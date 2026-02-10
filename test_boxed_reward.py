#!/usr/bin/env python3
"""
测试新的boxed格式reward函数
"""

import sys
import os
sys.path.append('/home/luzhenyan/verl_memagent')

from verl.utils.reward_score.segmented_reading import compute_score

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
    sys.exit(0 if success else 1)

