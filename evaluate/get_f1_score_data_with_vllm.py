import json
import os
import torch
import pandas as pd
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import pipeline, AutoTokenizer
from utils import md_to_dict_str, logger
import time


def get_f1_score_data_with_vllm(args):
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
            {"role": "user", "content": example['input']},
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
    prompts = dataset['text'].tolist()
    logger.info(f"输入示例: {prompts[0]}")

    # 初始化文本生成管道
    logger.info("初始化文本生成管道...")
    logger.info(prompts[0])
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=4096)
    evaluation_model = LLM(model_id, gpu_memory_utilization=0.9, max_seq_len_to_capture=4096,
                           tensor_parallel_size=len(os.getenv("CUDA_VISIBLE_DEVICES").split(',')))
    logger.info(f"输入示例: {prompts[0]}")
    outputs = evaluation_model.generate(prompts, sampling_params)
    logger.info(f"模型输出示例: {outputs[0]}")
    # 处理结果：提取生成内容并转换为字典
    results = [row[0]['text'][len(prompts[idx]):] for idx, row in enumerate(outputs)]
    results = [md_to_dict_str(row) for row in results]
    results = [json.loads(row) for row in results]
    logger.info(f"处理后输出示例: {results[0]}")

    # 保存结果
    output_path = Path(f"./data/{model_name}")
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / "results.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    logger.info(f"结果已保存至: {output_file}")
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
    get_f1_score_data_with_vllm(args)
    end_time = time.time()
    logger.info(f"耗时: {end_time - start_time}s")
