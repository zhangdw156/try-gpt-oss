import re
from loguru import logger as _logger
import pandas as pd
import os


def calculate_f1_score_from_csv(file_path: str) -> float:
    """
    读取指定的CSV文件，并根据TP, FP, FN列计算F1-score。

    Args:
        file_path (str): CSV文件的路径。

    Returns:
        float: 计算出的F1-score。如果TP+FP或TP+FN为0，则返回0.0。
    """
    try:
        # 使用pandas读取CSV文件
        df = pd.read_csv(file_path)

        # 确保必需的列存在
        required_columns = {'TP', 'FP', 'FN'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"CSV文件中缺少必需的列，需要包含: {', '.join(required_columns)}")

        # 计算所有行的TP, FP, FN的总和
        total_tp = df['TP'].sum()
        total_fp = df['FP'].sum()
        total_fn = df['FN'].sum()

        print(f"总 True Positives (TP): {total_tp}")
        print(f"总 False Positives (FP): {total_fp}")
        print(f"总 False Negatives (FN): {total_fn}")

        # 计算精确率 (Precision)
        # Precision = TP / (TP + FP)
        if total_tp + total_fp == 0:
            precision = 0.0
        else:
            precision = total_tp / (total_tp + total_fp)

        # 计算召回率 (Recall)
        # Recall = TP / (TP + FN)
        if total_tp + total_fn == 0:
            recall = 0.0
        else:
            recall = total_tp / (total_tp + total_fn)

        print(f"精确率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall): {recall:.4f}")

        # 计算F1-score
        # F1-score = 2 * (Precision * Recall) / (Precision + Recall)
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        return f1_score

    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
        return 0.0
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return 0.0


def get_logger():
    """
    自定义logger
    :return:
    """
    _logger.add("./logs/run.log", rotation="500 MB", encoding="utf-8")
    return _logger


def md_to_dict_str(text: str) -> str:
    """
    从Markdown文本中提取字典字符串
    :param text: Markdown文本
    :return: 提取的字典字符串，如果未找到则返回空字符串
    """
    pattern = r'(\{.*?\})'  # 匹配花括号内的内容
    match = re.search(pattern, text, re.DOTALL)  # 查找匹配项
    return match.group(1) if match else ''  # 返回匹配的字典字符串，否则返回空字符串


logger = get_logger()


def create_multisheet_excel_report(csv_list: list[str], output_excel_path: str):
    """
    读取指定目录下的所有CSV文件，将每个文件写入一个Excel工作表，
    并创建一个包含摘要信息的额外工作表。

    Args:
        csv_directory (str): 包含CSV文件的目录路径。
        output_excel_path (str): 输出的Excel文件路径。
    """
    # 用于存储摘要信息
    summary_data = []

    # 2. 使用 pd.ExcelWriter 作为上下文管理器，确保文件被正确保存和关闭
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:

        # 3. 遍历所有CSV文件
        for csv_file in csv_list:
            try:
                # 读取CSV文件到pandas DataFrame
                df = pd.read_csv(csv_file)

                # 从文件名创建一个简洁的sheet名称 (例如 'sales_data_2023.csv' -> 'sales_data_2023')
                sheet_name = csv_file.split('/')[-2] + '#' + csv_file.split('/')[-1].split('.')[0]
                # 确保sheet名长度不超过Excel限制（31个字符）
                sheet_name = sheet_name[:31]

                # 将DataFrame写入Excel的一个新sheet
                # index=False 参数可以防止pandas将DataFrame的索引写入Excel
                print(f"正在将 '{csv_file}' 写入 sheet '{sheet_name}'...")
                df.to_excel(writer, sheet_name=sheet_name, index=False)

                # 收集摘要信息（例如：文件名和行数）
                summary_data.append({
                    '来源文件': csv_file,
                    '工作表名称': sheet_name,
                    '行数': len(df),
                    '列数': len(df.columns)
                })

            except Exception as e:
                print(f"处理文件 '{csv_file}' 时出错: {e}")
                # 记录错误信息到摘要中
                summary_data.append({
                    '来源文件': csv_file,
                    '工作表名称': 'ERROR',
                    '行数': 0,
                    '列数': 0
                })

        # 4. 创建并写入摘要信息sheet
        print("正在创建摘要信息 sheet...")
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"\n处理完成！报告已保存至 '{output_excel_path}'")


if __name__ == '__main__':
    create_multisheet_excel_report(
        csv_list=[
            '../data/Qwen3-14B-bid/f1_score.csv',
            '../data/gpt-oss-20b-bid/f1_score.csv',
        ],
        output_excel_path="../data/f1_score_report.xlsx"
    )
