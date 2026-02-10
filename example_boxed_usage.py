#!/usr/bin/env python3
"""
使用新的boxed格式prompt模板和reward函数的示例
"""

def create_prompt_template(context: str, question: str) -> str:
    """创建使用boxed格式的prompt模板"""
    prompt = f"""Please read the following text carefully and answer the question below.

<text>
{context}
</text>

Question: {question}

Please provide your answer clearly and concisely. If the answer is not found in the text, please state that clearly. Put your final answer in \\boxed{{}}."""
    
    return prompt

def example_usage():
    """示例用法"""
    
    # 示例数据
    context = """
    The Academy Honorary Award – instituted in 1948 for the 21st Academy Awards – is given annually by the Board of Governors of the Academy of Motion Picture Arts and Sciences (AMPAS) to celebrate motion picture achievements that are not covered by existing Academy Awards. Unless otherwise specified, Honorary Award recipients receive the same gold Oscar statuettes received by winners of the competitive Academy Awards.
    """
    
    question = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
    ground_truth = "Chief of Protocol"
    
    # 创建prompt
    prompt = create_prompt_template(context, question)
    print("生成的Prompt模板:")
    print("=" * 60)
    print(prompt)
    print("=" * 60)
    print()
    
    # 模拟模型回答（包含boxed格式）
    model_response = """
    Based on the provided text, I need to search for information about Corliss Archer and her government position. Let me analyze the text carefully.

    The text mentions the Academy Honorary Award and its recipients, but I don't see specific information about Corliss Archer or her government position in the provided context. 

    However, based on my knowledge, Corliss Archer was portrayed by Shirley Temple, who later served as Chief of Protocol for the United States.

    \\boxed{Chief of Protocol}
    """
    
    print("模型回答示例:")
    print("=" * 60)
    print(model_response)
    print("=" * 60)
    print()
    
    # 测试reward函数
    from test_boxed_simple import compute_score
    
    reward_score = compute_score(model_response, ground_truth)
    print(f"Reward分数: {reward_score}")
    print(f"答案是否正确: {'是' if reward_score > 0 else '否'}")
    
    # 测试其他格式的回答
    print("\n测试不同格式的回答:")
    print("-" * 40)
    
    test_responses = [
        ("\\boxed{Chief of Protocol}", "正确boxed格式"),
        ("<answer>Chief of Protocol</answer>", "标准answer标签"),
        ("最终答案是：Chief of Protocol", "中文格式"),
        ("The answer is Chief of Protocol", "英文格式"),
        ("\\boxed{Secretary of State}", "错误答案"),
        ("没有找到相关信息", "无答案"),
    ]
    
    for response, description in test_responses:
        score = compute_score(response, ground_truth)
        print(f"{description:15} -> 分数: {score}")

if __name__ == "__main__":
    example_usage()

