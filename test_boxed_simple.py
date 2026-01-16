#!/usr/bin/env python3
"""
简单测试boxed格式答案提取
"""

import re

def _clean_answer(answer: str) -> str:
    """清理答案，移除多余的内容"""
    # 移除常见的结束标点和后续内容
    answer = re.sub(r'[。，！？\.!?]\s*.*$', '', answer)
    # 移除前后空白
    answer = answer.strip()
    # 移除引号
    answer = answer.strip('"\'""''')
    return answer

def _extract_final_answer(solution_str: str, method: str = "flexible") -> str:
    """从解决方案中提取最终答案，支持多种格式包括boxed格式"""
    # 优化：只检查最后的字符，提高性能
    if len(solution_str) > 1000:
        solution_str = solution_str[-1000:]
    
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

def _is_answer_correct(final_answer: str, ground_truth: str) -> bool:
    """检查答案是否正确"""
    if not final_answer:
        return False
    
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
    
    return accuracy >= 0.6

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

def compute_score(solution_str: str, ground_truth: str, method: str = "flexible", format_score: float = 0.0, score: float = 1.0) -> float:
    """计算奖励分数"""
    if not solution_str:
        return 0.0
    
    if not ground_truth:
        return 0.0
    
    # 提取最终答案
    final_answer = _extract_final_answer(solution_str, method=method)
    if not final_answer:
        return format_score
    
    # 检查答案是否正确
    if _is_answer_correct(final_answer, ground_truth):
        return score
    else:
        return format_score

def test_boxed_extraction():
    """测试boxed格式答案提取"""
    
    test_cases = [
        # 测试boxed格式
        {
            "solution": "经过分析，答案是42。\\boxed{42}",
            "ground_truth": "42",
            "expected": 1.0,
            "description": "标准boxed格式"
        },
        {
            "solution": "The answer is \\boxed{Chief of Protocol}",
            "ground_truth": "Chief of Protocol", 
            "expected": 1.0,
            "description": "英文boxed格式"
        },
        {
            "solution": "\\boxed{Greenwich Village, New York City}",
            "ground_truth": "Greenwich Village, New York City",
            "expected": 1.0,
            "description": "只有boxed格式"
        },
        {
            "solution": "\\boxed{ }",  # 空boxed
            "ground_truth": "42",
            "expected": 0.0,
            "description": "空boxed格式"
        },
        {
            "solution": "答案是42，\\boxed{42}",
            "ground_truth": "42",
            "expected": 1.0,
            "description": "混合格式"
        },
        # 测试其他格式
        {
            "solution": "<answer>42</answer>",
            "ground_truth": "42",
            "expected": 1.0,
            "description": "标准answer标签"
        },
        {
            "solution": "最终答案是：42",
            "ground_truth": "42",
            "expected": 1.0,
            "description": "中文答案格式"
        },
        {
            "solution": "The answer is 42",
            "ground_truth": "42",
            "expected": 1.0,
            "description": "英文答案格式"
        },
        # 测试错误答案
        {
            "solution": "答案是43。\\boxed{43}",
            "ground_truth": "42",
            "expected": 0.0,
            "description": "错误答案"
        },
        {
            "solution": "没有找到答案",
            "ground_truth": "42",
            "expected": 0.0,
            "description": "没有答案"
        }
    ]
    
    print("测试boxed格式reward函数")
    print("=" * 50)
    
    passed = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        try:
            result = compute_score(
                solution_str=case["solution"],
                ground_truth=case["ground_truth"],
                method="flexible"
            )
            
            success = abs(result - case["expected"]) < 1e-6
            status = "✓" if success else "✗"
            
            print(f"{i:2d}. {status} {case['description']}")
            print(f"    输入: {case['solution'][:50]}...")
            print(f"    期望: {case['expected']}, 实际: {result}")
            
            if success:
                passed += 1
            else:
                print(f"    ❌ 失败!")
            
            print()
            
        except Exception as e:
            print(f"{i:2d}. ✗ {case['description']} - 异常: {e}")
            print()
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    print(f"成功率: {passed/total*100:.1f}%")
    
    return passed == total

if __name__ == "__main__":
    success = test_boxed_extraction()
    exit(0 if success else 1)

