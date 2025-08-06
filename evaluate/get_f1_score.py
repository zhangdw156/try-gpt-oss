import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from dashscope import Generation
import argparse
from tqdm import tqdm
import time
import re
from utils import logger
from multiprocessing import Pool, cpu_count

# --- 全局配置和日志设置 ---
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("请在 .env 文件中设置 DASHSCOPE_API_KEY")

MODEL = "qwen-max"

# ... (全局配置部分)

# --- [新增] 定义所有评测中可能遇到的、需要报告的字段名 ---
# 将您数据集中的所有标准字段名都列在这里
# 这可以统一所有进程的输出格式
ALL_EVAL_FIELDS = [
    "项目名称", "公告类型", "行业分类", "发布时间", "预算金额",
    "采购人", "响应文件截至提交时间", "开标地址", "所在地区"
    # 如果还有其他可能的字段名变体，也统一映射到这里
]


# ...
# ==============================================================================
#  第一部分: 核心评测函数 (保持不变，但会被子进程调用)
# ==============================================================================

def get_field_similarity_score(value_a: str, value_b: str, field_name: str, context: str) -> int:
    """使用大语言模型判断两个字段值的语义相似度。"""
    # ... (此函数内容无需修改)
    value_a_str = str(value_a).strip()
    value_b_str = str(value_b).strip()

    if value_a_str == value_b_str:
        return 5

    prompt = f"""
请你扮演一个信息抽取和数据核对专家。
你的任务是判断【模型提取值】和【标准答案值】在给定的【上下文】和【字段名称】下，是否指代同一个信息。
请严格按照下面的评分标准，仅输出一个1到5之间的数字评分，不要有任何其他解释。

评分标准:
5: 完全相同或语义上完全等价（如：'联合国' vs 'U.N.'，'400000' vs '40万元'，'北京大学' vs '北大'）。
4: 高度相似，可以是缩写、别称或包含关系，但核心信息一致（如：'美国总统拜登' vs '拜登'）。
3: 部分相关，但信息不完整或范围不一致（如：'苹果公司' vs 'iPhone'）。
2: 轻微相关，属于同一范畴但并非指代同一事物（如：'苹果公司' vs '科技公司'）。
1: 完全不相关。

---
【上下文】: "{context[:2000]}"
【字段名称】: "{field_name}"
【模型提取值】: "{value_a_str}"
【标准答案值】: "{value_b_str}"
---

评分 (仅输出数字1/2/3/4/5):
"""
    try:
        # 在高并发下，API可能会有限流，适当的延时和重试机制是好的实践
        time.sleep(0.1)  # 稍微减少延时，因为是并行
        response = Generation.call(
            model=MODEL,
            api_key=DASHSCOPE_API_KEY,
            prompt=prompt,
            result_format='text'
        )
        if response.status_code == 200:
            result_text = response.output.text
            match = re.search(r'\d+', result_text)
            if match:
                score = int(match.group(0))
                return max(1, min(5, score))
            else:
                # 在多进程中，日志可能交叉，返回错误信息是更好的方式
                # logger.warning(...)
                return 1
        else:
            return 1
    except Exception:
        return 1


def clean_value(value):
    """清理值，去除首尾空格"""
    if isinstance(value, str):
        return value.strip()
    return value


def evaluate_json_extraction(model_data: dict, ground_truth_data: dict, context: str, similarity_threshold: int = 4):
    """
    评估模型抽取的字典与标准答案字典的差异。
    此版本使用标准的TP, FP, FN计算逻辑，并基于固定的字段列表生成报告。
    """
    # 1. 筛选出各自的有效预测集合（这部分不变）
    gt_set = {k: v for k, v in ground_truth_data.items() if v is not None and str(v).lower() != 'none' and v != ''}
    model_set = {k: v for k, v in model_data.items() if v is not None and str(v).lower() != 'none' and v != ''}

    # 2. 计算 TP (这部分不变)
    tp = 0
    common_keys = set(gt_set.keys()) & set(model_set.keys())
    for key in common_keys:
        # ... (内部的语义判断逻辑不变) ...
        # [代码省略，与上一版相同]
        # ...
        score = 1
        if "金额" in key:
            try:
                gt_num_str = re.sub(r'[^\d.]', '', str(gt_set[key]))
                model_num_str = re.sub(r'[^\d.]', '', str(model_set[key]))
                if gt_num_str and model_num_str and abs(float(gt_num_str) - float(model_num_str)) < 1e-6:
                    score = 5
            except (ValueError, TypeError):
                score = get_field_similarity_score(str(model_set[key]), str(gt_set[key]), key, context)
        else:
            score = get_field_similarity_score(str(model_set[key]), str(gt_set[key]), key, context)
        if score >= similarity_threshold:
            tp += 1

    # 3. 根据标准公式计算 FP 和 FN (这部分不变)
    fp = len(model_set) - tp
    fn = len(gt_set) - tp

    # --- [修改] 4. 生成固定的、有序的字段级报告 ---
    field_results = {}
    # 只遍历我们预定义的标准字段
    for key in ALL_EVAL_FIELDS:
        gt_value = ground_truth_data.get(key)
        model_value = model_data.get(key)

        is_gt_empty = gt_value is None or str(gt_value).lower() == 'none' or gt_value == ''
        is_model_empty = model_value is None or str(model_value).lower() == 'none' or model_value == ''

        if is_gt_empty and is_model_empty:
            field_results[key] = 'OK (both empty)'
        elif not is_gt_empty and not is_model_empty:
            # 这里可以复用上面的相似度判断，但为了简化，我们直接标记状态
            # 检查这个键是否在TP的范畴内
            if key in common_keys and get_field_similarity_score(str(model_value), str(gt_value), key,
                                                                 context) >= similarity_threshold:
                field_results[key] = 'OK (Matched)'
            else:
                field_results[key] = 'Mismatch'
        elif not is_gt_empty and is_model_empty:
            field_results[key] = 'FN (Missed)'
        elif is_gt_empty and not is_model_empty:
            # 这里需要注意，一个模型幻觉出的字段如果不在ALL_EVAL_FIELDS里，就不会被报告
            # 这是合理的，因为我们只关心标准字段的表现
            field_results[key] = 'FP (Hallucinated)'

    return {'details': field_results, 'tp': tp, 'fp': fp, 'fn': fn}


# ==============================================================================
#  第二部分: 新增的工作函数 (Worker Function)
# ==============================================================================

def process_line(task_data):
    """
    这是一个工作函数，由每个子进程执行，负责处理单行数据的评测。
    当模型输出无效时，会进行惩罚性计分而不是跳过。
    """
    line_number, ds_line, gt_line, model_line, threshold = task_data

    # 1. 解析标准答案和上下文
    try:
        ds_item = json.loads(ds_line)
        ground_truth_data = json.loads(gt_line)
        if not isinstance(ground_truth_data, dict):
            return {'line_number': line_number, 'error': "标准答案文件行不是一个有效的JSON对象"}
    except json.JSONDecodeError as e:
        return {'line_number': line_number, 'error': f"标准答案JSON解析失败: {e}"}

    context = ds_item.get('input', '')
    if not context:
        return {'line_number': line_number, 'error': "数据集中找不到 'input' 字段"}

    # 2. 解析模型输出，并处理无效情况
    try:
        model_data = json.loads(model_line)
        if not isinstance(model_data, dict):
            # 如果模型输出不是字典 (例如是 null), 将 model_data 视为空字典进行惩罚性计分
            model_data = {}
    except json.JSONDecodeError:
        # 如果模型输出是无效JSON (例如空字符串或乱码), 也视为空字典
        model_data = {}

    # 3. 进行评测
    # 此时 model_data 要么是有效的字典，要么是空字典，可以直接传入评测函数
    eval_result = evaluate_json_extraction(model_data, ground_truth_data, context, threshold)

    # 4. 准备返回结果
    record = {
        'line_number': line_number,
        'TP': eval_result['tp'],
        'FP': eval_result['fp'],
        'FN': eval_result['fn'],
    }
    # 即使模型输出无效，也要记录每个字段的评测结果（大部分会是FN）
    record.update({f'field_{k}': v for k, v in eval_result['details'].items()})

    # [可选] 增加一个字段标记模型输出是否有效，方便后续分析
    if not model_data:
        record['model_output_valid'] = False
    else:
        record['model_output_valid'] = True

    return record


# ==============================================================================
#  第三部分: 修改后的主程序逻辑
# ==============================================================================
def process_files(lines_dataset, lines_gt, lines_model):
    # 获取三个文件的行数
    len_dataset = len(lines_dataset)
    len_gt = len(lines_gt)
    len_model = len(lines_model)

    # 检查行数是否匹配
    if not (len_dataset == len_gt == len_model):
        logger.warning(f"输入文件行数不匹配! 分别为: dataset={len_dataset}, gt={len_gt}, model={len_model}")

        # 取最短的行数
        min_length = min(len_dataset, len_gt, len_model)
        logger.warning(f"统一为最短行数: {min_length}")

        # 截断到最短行数
        lines_dataset = lines_dataset[:min_length]
        lines_gt = lines_gt[:min_length]
        lines_model = lines_model[:min_length]

        # 验证截断后行数是否一致
        if not (len(lines_dataset) == len(lines_gt) == len(lines_model) == min_length):
            logger.error("截断后行数仍然不匹配，无法继续处理")
            return None

    # 行数匹配后继续处理逻辑
    logger.info(f"文件行数校验通过，共 {len(lines_dataset)} 行")
    # ... 后续处理代码 ...

    return lines_dataset, lines_gt, lines_model


def main(args):
    # -- 1. 读取所有输入文件 --
    try:
        with open(args.dataset_file, 'r', encoding='utf-8') as f_ds, \
                open(args.ground_truth_file, 'r', encoding='utf-8') as f_gt, \
                open(args.model_output_file, 'r', encoding='utf-8') as f_model:

            lines_dataset = f_ds.readlines()
            lines_gt = f_gt.readlines()
            lines_model = f_model.readlines()

    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}. 请检查文件路径。")
        return

    # -- 2. 校验文件行数 --
    if not (len(lines_dataset) == len(lines_gt) == len(lines_model)):
        logger.warning("输入文件行数不匹配!")
        lines_dataset, lines_gt, lines_model = process_files(lines_dataset, lines_gt, lines_model)

    # -- 3. 准备任务列表 --
    tasks = []
    for i, (ds_line, gt_line, model_line) in enumerate(zip(lines_dataset, lines_gt, lines_model)):
        tasks.append((i + 1, ds_line, gt_line, model_line, args.threshold))

    # -- 4. 使用多进程池进行评测 --
    all_results_details = []
    logger.info(f"使用 {args.num_processes} 个进程开始评测 {len(tasks)} 条数据...")

    with Pool(processes=args.num_processes) as pool:
        # 使用 pool.imap 以便能够实时更新进度条
        results_iterator = pool.imap(process_line, tasks)

        for result in tqdm(results_iterator, total=len(tasks)):
            if 'error' in result:
                logger.warning(f"跳过第 {result['line_number']} 行，原因: {result['error']}")
            else:
                all_results_details.append(result)

    # -- 5. 汇总统计结果 --
    total_tp, total_fp, total_fn = 0, 0, 0
    for result in all_results_details:
        total_tp += result.get('TP', 0)
        total_fp += result.get('FP', 0)
        total_fn += result.get('FN', 0)

    # -- 6. 计算并打印最终指标 --
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    logger.info("=" * 30)
    logger.info("评测结果汇总")
    logger.info("=" * 30)
    logger.info(f"相似度阈值 (Threshold): {args.threshold}")
    logger.info(f"总计 True Positives (TP - 字段匹配正确): {total_tp}")
    logger.info(f"总计 False Positives (FP - 模型多提/提错): {total_fp}")
    logger.info(f"总计 False Negatives (FN - 模型漏提/提错): {total_fn}")
    logger.info(f"整体精确率 (Precision): {precision:.4f}")
    logger.info(f"整体召回率 (Recall): {recall:.4f}")
    logger.info(f"整体 F1-Score: {f1_score:.4f}")
    logger.info("=" * 30)

    # -- 7. 保存详细结果 --
    if all_results_details:
        # --- [新增] 定义最终CSV文件的列顺序 ---
        # 基础列
        base_columns = ['line_number', 'TP', 'FP', 'FN', 'model_output_valid']
        # 字段详情列
        field_columns = [f'field_{field}' for field in ALL_EVAL_FIELDS]
        # 最终的列顺序
        final_columns_order = base_columns + field_columns

        results_df = pd.DataFrame(all_results_details).sort_values(by='line_number')

        # --- [修改] 使用 reindex 来确保所有行都有相同的列，并按指定顺序排列 ---
        # 对于在 all_results_details 中不存在的列，会自动用 NaN 填充
        results_df = results_df.reindex(columns=final_columns_order)

        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
        logger.info(f"详细评测结果已保存至: {args.output_file}")
    else:
        logger.warning("没有成功处理任何数据，不生成结果文件。")


if __name__ == '__main__':
    # 获取CPU核心数作为默认进程数参考
    default_processes = min(20, cpu_count())

    parser = argparse.ArgumentParser(description="使用LLM多进程评测结构化JSON抽取任务")
    parser.add_argument("--dataset_file", type=str,
                        help="包含 instruction 和 input 的 .jsonl 数据集文件路径",
                        default="../dataset/bid-announcement-zh-v1.0.jsonl")
    parser.add_argument("--ground_truth_file", type=str,
                        help="包含标准答案字典的文本文件路径 (每行一个字典)",
                        default="../data/gt.txt")
    parser.add_argument("--model_output_file", type=str,
                        help="包含模型输出字典的文本文件路径 (每行一个字典)",
                        default="../data/gpt-oss-20b-bid/results.txt")
    parser.add_argument("--output_file", type=str,
                        default="../data/gpt-oss-20b-bid/f1_score.csv",
                        help="输出评测结果的CSV文件路径")
    parser.add_argument("--threshold", type=int, default=4, choices=[1, 2, 3, 4, 5],
                        help="判断为匹配成功的相似度分数阈值")
    parser.add_argument("--num_processes", type=int, default=20,
                        help=f"用于评测的进程数 (建议不超过CPU核心数: {cpu_count()})")

    args = parser.parse_args()
    main(args)
