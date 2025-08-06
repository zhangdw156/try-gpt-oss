import re
from loguru import logger as _logger


def get_logger():
    _logger.add("../logs/run.log")
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
