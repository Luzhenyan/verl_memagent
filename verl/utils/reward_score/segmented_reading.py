"""
Segmented Reading Reward Function for VERL
基于已有奖励函数的模式，为分段阅读任务设计
"""

import re
import logging
from typing import Dict, Any, List, Union

logger = logging.getLogger(__name__)

_SOLUTION_CLIP_CHARS = 1000  # 分段阅读回答可能更长


def compute_score(
    solution_str: str,
    ground_truth: Union[str, Dict[str, Any]],
    method: str = "flexible",
    format_score: float = 0.0,
    score: float = 1.0,
    **kwargs
) -> float:
    """
    计算分段阅读任务的奖励分数
    兼容现有框架的接口设计
    
    Args:
        solution_str: 模型的解决方案字符串（多轮对话的完整输出）
        ground_truth: 正确答案，可以是字符串或包含target字段的字典
        method: 提取方法，'strict' 或 'flexible'
        format_score: 格式正确但答案错误时的分数
        score: 答案正确时的分数
        **kwargs: 其他参数
    
    Returns:
        float: 奖励分数
    """
    
    if not solution_str:
        return 0.0
    
    # 处理不同格式的ground_truth
    if isinstance(ground_truth, dict):
        if "target" in ground_truth:
            target_answer = ground_truth["target"]
        elif "answer" in ground_truth:
            target_answer = ground_truth["answer"]
        else:
            # 如果字典中没有预期的键，尝试获取第一个值
            target_answer = next(iter(ground_truth.values())) if ground_truth else ""
    else:
        target_answer = ground_truth
    
    if not target_answer:
        return 0.0
    
    # 提取最终答案
    final_answer = _extract_final_answer(solution_str, method=method)
    if not final_answer:
        return format_score
    
    # 检查答案是否正确
    if _is_answer_correct(final_answer, target_answer):
        return score
    else:
        return format_score


def _is_answer_correct(final_answer: str, ground_truth: Union[str, List[str]]) -> bool:
    """检查答案是否正确，支持多个正确答案"""
    if not final_answer:
        return False
    
    # 处理多个正确答案的情况
    if isinstance(ground_truth, list):
        truth_list = ground_truth
    elif isinstance(ground_truth, str):
        truth_list = [ground_truth]
    else:
        return False
    
    # 对每个可能的正确答案进行检查
    for truth in truth_list:
        if not truth:
            continue
            
        if _single_answer_check(final_answer, truth):
            return True
    
    return False


def _single_answer_check(final_answer: str, ground_truth: str) -> bool:
    """检查单个答案是否匹配"""
    # 规范化答案
    answer_norm = _normalize_answer(final_answer)
    truth_norm = _normalize_answer(ground_truth)
    
    if not answer_norm or not truth_norm:
        return False
    
    # 1. 精确匹配
    if answer_norm == truth_norm:
        return True
    
    # 2. 包含匹配（适用于短答案）
    if len(truth_norm) <= 20:  # 短答案用包含匹配
        if truth_norm in answer_norm or answer_norm in truth_norm:
            return True
    
    # 3. 数字匹配（提取数字进行比较）
    answer_numbers = re.findall(r'-?\d+\.?\d*', answer_norm)
    truth_numbers = re.findall(r'-?\d+\.?\d*', truth_norm) 
    if answer_numbers and truth_numbers:
        # 如果两者都包含数字，比较数字
        try:
            answer_num = float(answer_numbers[0])
            truth_num = float(truth_numbers[0])
            if abs(answer_num - truth_num) < 1e-6:
                return True
        except ValueError:
            pass
    
    # 4. 关键词匹配（至少60%的关键词匹配）
    answer_words = set(re.findall(r'\w+', answer_norm))
    truth_words = set(re.findall(r'\w+', truth_norm))
    
    if not truth_words:
        return False
    
    overlap = len(answer_words.intersection(truth_words))
    accuracy = overlap / len(truth_words)
    
    return accuracy >= 0.6  # 提高阈值以减少假阳性


def _normalize_answer(answer: str) -> str:
    """规范化答案字符串"""
    if not answer:
        return ""
    
    # 转换为小写
    answer = answer.lower().strip()
    
    # 移除标点符号
    answer = re.sub(r'[^\w\s\u4e00-\u9fff]', '', answer)
    
    # 移除多余的空格
    answer = re.sub(r'\s+', ' ', answer)
    
    return answer.strip()


def _extract_final_answer(solution_str: str, method: str = "flexible") -> str:
    """从解决方案中提取最终答案，支持多种格式包括boxed格式"""
    # 优化：只检查最后的字符，提高性能
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]
    
    # 1. 首先尝试查找boxed格式的答案（优先级最高）
    boxed_patterns = [
        r"\\boxed\{([^}]+)\}",  # \boxed{answer}
        r"\\boxed\{([^}]*)\}",  # \boxed{} 空boxed
        r"\\boxed\s*\{([^}]*)\}",  # \boxed {answer} 带空格
        r"\\boxed\s*\{([^}]*)\s*\}",  # \boxed { answer } 带空格
    ]
    
    for pattern in boxed_patterns:
        matches = re.findall(pattern, solution_str, re.IGNORECASE | re.DOTALL)
        if matches:
            # 取最后一个匹配的答案
            answer = matches[-1].strip()
            if answer:
                return _clean_answer(answer)
    
    # 2. 查找标准答案标签
    answer_tag_patterns = [
        r"<answer>\s*(.*?)\s*</answer>",
        r"<ANSWER>\s*(.*?)\s*</ANSWER>",
    ]
    
    for pattern in answer_tag_patterns:
        matches = re.findall(pattern, solution_str, re.IGNORECASE | re.DOTALL)
        if matches:
            # 取最后一个匹配的答案
            answer = matches[-1].strip()
            if answer:
                return _clean_answer(answer)
    
    # 3. 查找明确的答案标记模式
    explicit_patterns = [
        r"最终答案[：:]\s*(.+)",
        r"答案是[：:]\s*(.+)", 
        r"答案[：:]\s*(.+)",
        r"Final answer[：:]\s*(.+)",
        r"Answer[：:]\s*(.+)",
        r"The answer is[：:]\s*(.+)",
        r"Based on.*?(?:answer is|答案是)\s*(.+)",
        r"Therefore.*?(?:answer is|答案是)\s*(.+)",
        r"In conclusion.*?(?:answer is|答案是)\s*(.+)",
    ]
    
    for pattern in explicit_patterns:
        match = re.search(pattern, solution_str, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer:
                return _clean_answer(answer)
    
    # 如果是strict模式，到这里就返回空
    if method == "strict":
        return ""
    
    # 4. flexible模式：尝试从最后几行提取答案
    lines = solution_str.strip().split('\n')
    for line in reversed(lines[-5:]):  # 检查最后5行
        line = line.strip()
        if line and not _is_system_message(line):
            cleaned_line = _clean_answer(line)
            if len(cleaned_line) >= 1:  # 答案至少1个字符
                return cleaned_line
    
    return ""


def _clean_answer(answer: str) -> str:
    """清理答案，移除多余的内容"""
    # 移除常见的结束标点和后续内容
    answer = re.sub(r'[。，！？\.!?]\s*.*$', '', answer)
    # 移除前后空白
    answer = answer.strip()
    # 移除引号
    answer = answer.strip('"\'""''')
    return answer


def _is_system_message(line: str) -> bool:
    """检查是否是系统消息"""
    system_prefixes = [
        "User:", "user:", "USER:",
        "Assistant:", "assistant:", "ASSISTANT:", 
        "Tool:", "tool:", "TOOL:",
        "System:", "system:", "SYSTEM:",
        "Human:", "human:", "HUMAN:",
        "AI:", "ai:",
    ]
    return any(line.startswith(prefix) for prefix in system_prefixes)
