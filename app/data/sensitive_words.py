"""敏感词词典 - 用于内容安全过滤"""

# 定义敏感词列表（可根据需要扩展）
SENSITIVE_WORDS = [
    # 政治相关
    "共产党", "中国政府", "中华人民共和国", "政府", "总统", "总理",
    # 色情相关
    "色", "黄", "裸", "性", "床", "脱", "内裤", "胸", "屁股",
    # 暴力相关
    "杀", "死", "打", "砸", "暴力", "凶器", "血", "刀", "枪",
    # 辱骂相关
    "傻逼", "傻bi", "SB", "sb", "渣男", "渣女", "神经病", "疯子",
    # 违法相关
    "贩毒", "走私", "拐卖", "诈骗", "违法", "犯罪",
    # 其他敏感词
    "自杀", "跳楼", "自残",
]

# 构建正则表达式模式 - 匹配任何敏感词
import re


def build_pattern(words: list[str]) -> re.Pattern:
    """
    构建敏感词匹配的正则表达式

    Args:
        words: 敏感词列表

    Returns:
        编译后的正则表达式对象
    """
    # 对每个词进行转义，避免特殊字符冲突
    escaped_words = [re.escape(word) for word in words]
    # 构建正则模式：匹配任何敏感词
    pattern = "|".join(escaped_words)
    return re.compile(pattern, re.IGNORECASE)


# 全局编译的正则表达式
SENSITIVE_PATTERN = build_pattern(SENSITIVE_WORDS)


def contains_sensitive_word(text: str) -> tuple[bool, str | None]:
    """
    检查文本是否包含敏感词

    Args:
        text: 待检查的文本

    Returns:
        (是否包含敏感词, 匹配到的敏感词或None)
    """
    match = SENSITIVE_PATTERN.search(text)
    if match:
        return True, match.group()
    return False, None


def clean_text(text: str) -> str:
    """
    替换文本中的敏感词为星号

    Args:
        text: 原始文本

    Returns:
        替换后的文本
    """
    return SENSITIVE_PATTERN.sub("*" * 3, text)
