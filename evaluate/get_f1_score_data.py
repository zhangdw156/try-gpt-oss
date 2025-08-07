import json
import os
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig

from utils import md_to_dict_str, logger


def process_batch_with_retry(pipe, batch_prompts: list, max_retries_for_single: int = 2):
    """
    使用递归和重试机制处理一个批次。

    如果失败，它会尝试将批次拆分为更小的块。
    如果一个单独的项失败，它会重试几次，然后才放弃。

    Args:
        pipe: The transformers pipeline.
        batch_prompts (list): The list of prompts to process.
        max_retries_for_single (int): Number of retries for a single failing prompt.

    Returns:
        list: The output from the pipeline, structured as if it succeeded.
    """
    try:
        # 1. 尝试正常处理整个批次
        logger.info(f"正在尝试处理批次，大小: {len(batch_prompts)}")
        outputs = pipe(
            batch_prompts,
            max_new_tokens=2048,
            temperature=0.7,
            do_sample=True,
            batch_size=4
        )
        logger.info(f"批次 (大小: {len(batch_prompts)}) 处理成功。")
        return outputs

    except Exception as e:
        # 2. 如果失败，记录错误并开始重试逻辑
        logger.warning(f"批次 (大小: {len(batch_prompts)}) 推理失败: {e}")

        # 3. 检查批次大小
        if len(batch_prompts) > 1:
            # 3a. 如果批次大小 > 1, 拆分并递归
            logger.info("批次大小 > 1，将拆分为两半进行重试...")
            mid_point = len(batch_prompts) // 2
            first_half = batch_prompts[:mid_point]
            second_half = batch_prompts[mid_point:]

            # 递归调用
            outputs_first = process_batch_with_retry(pipe, first_half, max_retries_for_single)
            outputs_second = process_batch_with_retry(pipe, second_half, max_retries_for_single)

            # 合并成功的结果
            return outputs_first + outputs_second

        else:  # len(batch_prompts) == 1
            # 3b. 如果批次大小为 1, 进入重试循环
            logger.error("单个 prompt 推理失败，将进行最多 {max_retries_for_single} 次重试...")
            for i in range(max_retries_for_single):
                logger.info(f"重试 {i + 1}/{max_retries_for_single}...")
                time.sleep(2)  # 在重试前短暂等待，有时有助于解决瞬时问题
                try:
                    outputs = pipe(
                        batch_prompts,
                        max_new_tokens=2048,
                        temperature=0.7,
                        do_sample=True,
                        batch_size=4
                    )
                    logger.info("单个 prompt 在重试后成功！")
                    return outputs
                except Exception as retry_e:
                    logger.error(f"重试 {i + 1} 失败: {retry_e}")

            # 4. 如果所有重试都失败了，使用最终回退机制
            logger.critical("所有重试均失败，将使用回退机制。")
            return [[{'generated_text': batch_prompts[0]}]]


def get_f1_score_data(args):
    """
    生成用于计算F1分数的数据，包括模型预测结果

    参数:
        args: 包含配置参数的对象，应包含dataset_id等必要属性
    """
    # 优先使用环境变量配置，其次使用参数或默认值
    cuda_devices = os.getenv('CUDA_VISIBLE_DEVICES', '0,1,6,7')
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices

    # 模型路径：环境变量 > args参数 > 默认值
    model_id = getattr(args, 'model_id',
                       "/data/finetuning/finetuned-model/gpt-oss-20b-bid")

    # 数据集路径：优先使用args参数
    dataset_id = getattr(args, 'dataset_id', './dataset/bid-announcement-zh-v1.0.jsonl')

    # 加载数据集
    logger.info(f"加载数据集: {dataset_id}")
    dataset = pd.read_json(dataset_id, lines=True)

    # 加载分词器
    logger.info(f"加载分词器: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 提取模型名称
    model_name = model_id.split('/')[-1]
    logger.info(f"使用模型: {model_name}")

    # 格式化函数：将样本转换为聊天模板格式
    def formatting(example: dict):
        messages = [
            {"role": "system", "content": example['instruction']},
            {"role": "user", "content": example['input'][:4096]},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    # 应用格式化
    dataset['text'] = dataset.apply(formatting, axis=1)
    if dataset.loc[0, 'text'].find("Reasoning: medium"):
        dataset['text'] = dataset['text'].apply(
            lambda x: x.replace("Reasoning: medium", "Reasoning: low")
        )
    # 找到最长的text
    max_length = dataset['text'].str.len().max()
    longest_text = dataset[dataset['text'].str.len() == max_length]['text'].iloc[0]

    logger.info(f"最长文本长度: {max_length}")
    logger.info(f"最长文本内容:\n{longest_text}")

    # 找到最短的text
    min_length = dataset['text'].str.len().min()
    shortest_text = dataset[dataset['text'].str.len() == min_length]['text'].iloc[0]

    logger.info(f"最短文本长度: {min_length}")
    logger.info(f"最短文本内容:\n{shortest_text}")

    # 计算平均文本长度
    avg_length = dataset['text'].str.len().mean()  # 核心代码：求长度的平均值
    logger.info(f"平均文本长度: {round(avg_length, 2)}")  # 保留两位小数，更易读

    prompts = dataset['text'].tolist()
    logger.info(f"输入示例: {prompts[0]}")

    # 初始化文本生成管道
    logger.info("初始化文本生成管道...")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={
            # "quantization_config": quantization_config,
            "device_map": "auto"
        },
    )

    # 访问分词器, 设置 padding_side 为 'left'
    pipe.tokenizer.padding_side = 'left'

    # 生成预测结果（取前10个样本）
    logger.info("开始生成预测结果...")
    sample_size = getattr(args, 'sample_size', 10)
    logger.info(f"采样数量: {sample_size}")
    prompts = prompts[:sample_size]
    batch_size = 20
    output_path = Path(f"./data/{model_name}")
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "results.txt"

    # 重要：在开始循环前，以写入模式('w')打开文件一次，以清空之前运行的内容。
    # 如果您希望在多次运行脚本时持续追加，可以移除这一行。
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("")  # 创建或清空文件

    # 2. 使用tqdm创建进度条并按批次循环
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    logger.info(f"批次数量: {num_batches}")
    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing Batches", total=num_batches):

        # 获取当前批次的prompts
        batch_prompts = prompts[i: i + batch_size]

        # 如果当前批次为空，则跳过
        if not batch_prompts:
            continue

        logger.info(f"正在处理批次 {i // batch_size + 1}/{num_batches}，包含 {len(batch_prompts)} 个样本。")

        # 3. 对当前批次进行推理
        # 错误处理：将模型推理放入try-except块，以防批次处理失败
        # === 调用我们新的、健壮的辅助函数 ===
        outputs = process_batch_with_retry(pipe, batch_prompts)

        # 4. 处理当前批次的结果
        # 错误处理：对每条结果进行单独处理，防止单个格式错误导致整个批次失败
        processed_results = []
        for idx, row in enumerate(outputs):
            try:
                # 1. 提取生成的内容
                generated_part = row[0]['generated_text'][len(batch_prompts[idx]):]

                # 2. 转换为字典字符串并解析为JSON
                dict_str = md_to_dict_str(generated_part)
                json_obj = json.loads(dict_str)

                # 3. 如果成功，添加解析好的对象
                processed_results.append(json_obj)

            except json.JSONDecodeError as e:
                # 4a. 如果JSON解析失败，记录日志并添加错误对象
                error_message = f"JSON解析失败: {e}"
                logger.warning(f"{error_message}。原始文本: '{dict_str[:150]}...'")
                processed_results.append({
                    'error': 'JSONDecodeError',
                    'message': error_message,
                    'raw_output': dict_str  # 将导致错误的原始字符串也记录下来，便于调试
                })

            except Exception as e:
                # 4b. 如果发生其他任何异常，也记录日志并添加错误对象
                error_message = f"处理单条结果时发生未知错误: {e}"
                logger.error(error_message)
                processed_results.append({
                    'error': 'UnknownProcessingError',
                    'message': error_message,
                    'raw_output': row[0]['generated_text']  # 记录模型最原始的输出
                })

        logger.info(f"批次 {i // batch_size + 1} 处理完成，成功解析 {len(processed_results)} 条结果。")
        if processed_results:
            logger.info(f"输出示例: {processed_results[0]}")

        # 5. 将处理好的结果追加到文件
        # 使用追加模式 'a'
        with open(output_file, "a", encoding="utf-8") as f:
            for res in processed_results:
                results.append(res)
                f.write(json.dumps(res, ensure_ascii=False) + '\n')

    logger.info(f"所有批次处理完毕！结果已保存至 {output_file}")
    return results


if __name__ == "__main__":
    # 示例参数配置
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成F1评分数据')
    parser.add_argument('--dataset_id', type=str,
                        default='./dataset/bid-announcement-zh-v1.0.jsonl',
                        help='数据集路径')
    parser.add_argument('--model_id', type=str,
                        default='/data/finetuning/finetuned-model/gpt-oss-20b-bid',
                        help='模型路径（优先使用环境变量MODEL_ID）')
    parser.add_argument('--sample_size', type=int, default=10,
                        help='生成样本数量')

    args = parser.parse_args()

    start_time = time.time()
    # 调用函数
    get_f1_score_data(args)
    end_time = time.time()
    logger.info(f"耗时: {end_time - start_time}s")
