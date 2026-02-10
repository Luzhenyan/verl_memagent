#!/usr/bin/env python3
"""
测试分段阅读Agent Loop
验证Agent Loop是否能正确使用文件操作工具
"""

import sys
import os
import asyncio
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from verl.experimental.agent_loop.segmented_reading_agent_loop import SegmentedReadingAgentLoop

class MockTokenizer:
    """模拟tokenizer"""
    def encode(self, text):
        return [1, 2, 3, 4, 5]  # 模拟token ids
    
    def decode(self, token_ids):
        return "模拟解码的文本"

class MockProcessor:
    """模拟processor"""
    def __call__(self, *args, **kwargs):
        return {"input_ids": [1, 2, 3, 4, 5]}

class MockServerManager:
    """模拟server manager"""
    pass

class MockTrainerConfig:
    """模拟trainer config"""
    def __init__(self, config):
        self.config = config

async def test_agent_loop():
    """测试Agent Loop"""
    print("=== 测试 SegmentedReadingAgentLoop ===")
    
    # 创建模拟组件
    tokenizer = MockTokenizer()
    processor = MockProcessor()
    server_manager = MockServerManager()
    config = {
        "max_segments_to_read": 3,
        "max_turns": 5
    }
    trainer_config = MockTrainerConfig(config)
    
    # 创建Agent Loop实例
    agent_loop = SegmentedReadingAgentLoop(trainer_config, server_manager, tokenizer, processor)
    
    # 测试参数
    sampling_params = {
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    # 测试数据
    kwargs = {
        "document_file": "data/triviaqa_docs/document_0.json"
    }
    
    print(f"测试文档文件: {kwargs['document_file']}")
    print(f"配置: {config}")
    
    # 运行Agent Loop
    try:
        result = await agent_loop.run(sampling_params, **kwargs)
        
        print("\n=== 测试结果 ===")
        print(f"Prompt IDs: {result.prompt_ids}")
        print(f"Response IDs: {result.response_ids}")
        print(f"Num turns: {result.num_turns}")
        print(f"指标: {result.metrics}")
        
        # 检查结果
        if result.num_turns > 0:
            print("✅ Agent Loop执行成功")
            return True
        else:
            print("❌ Agent Loop执行失败")
            return False
            
    except Exception as e:
        print(f"❌ Agent Loop执行异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_agent_loop_with_nonexistent_file():
    """测试Agent Loop处理不存在的文件"""
    print("\n=== 测试处理不存在的文件 ===")
    
    # 创建模拟组件
    tokenizer = MockTokenizer()
    processor = MockProcessor()
    server_manager = MockServerManager()
    config = {
        "max_segments_to_read": 3,
        "max_turns": 5
    }
    trainer_config = MockTrainerConfig(config)
    
    # 创建Agent Loop实例
    agent_loop = SegmentedReadingAgentLoop(trainer_config, server_manager, tokenizer, processor)
    
    # 测试参数
    sampling_params = {
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    # 测试数据 - 不存在的文件
    kwargs = {
        "document_file": "data/triviaqa_docs/nonexistent.json"
    }
    
    print(f"测试不存在的文件: {kwargs['document_file']}")
    
    # 运行Agent Loop
    try:
        result = await agent_loop.run(sampling_params, **kwargs)
        
        print(f"Prompt IDs: {result.prompt_ids}")
        print(f"Response IDs: {result.response_ids}")
        print(f"Num turns: {result.num_turns}")
        print(f"指标: {result.metrics}")
        
        # 检查结果
        if result.num_turns == 0:
            print("✅ 正确处理了不存在的文件")
            return True
        else:
            print("❌ 没有正确处理不存在的文件")
            return False
            
    except Exception as e:
        print(f"❌ 处理不存在的文件时发生异常: {str(e)}")
        return False

async def test_tool_initialization():
    """测试工具初始化"""
    print("\n=== 测试工具初始化 ===")
    
    # 创建模拟组件
    tokenizer = MockTokenizer()
    processor = MockProcessor()
    server_manager = MockServerManager()
    config = {
        "max_segments_to_read": 3,
        "max_turns": 5
    }
    trainer_config = MockTrainerConfig(config)
    
    # 创建Agent Loop实例
    agent_loop = SegmentedReadingAgentLoop(trainer_config, server_manager, tokenizer, processor)
    
    # 检查工具是否正确初始化
    expected_tools = ["read_segment_file", "write_summary_file", "read_summary_file", "get_document_info"]
    
    for tool_name in expected_tools:
        if tool_name in agent_loop.tools:
            print(f"✅ 工具 {tool_name} 已初始化")
        else:
            print(f"❌ 工具 {tool_name} 未初始化")
            return False
    
    print("✅ 所有工具都已正确初始化")
    return True

async def main():
    """主函数"""
    print("开始测试分段阅读Agent Loop...")
    print("=" * 60)
    
    # 确保测试目录存在
    os.makedirs("data/triviaqa_docs", exist_ok=True)
    
    # 运行各项测试
    results = []
    
    # 测试工具初始化
    results.append(await test_tool_initialization())
    
    # 测试处理不存在的文件
    results.append(await test_agent_loop_with_nonexistent_file())
    
    # 测试正常执行（如果文档文件存在）
    if os.path.exists("data/triviaqa_docs/document_0.json"):
        results.append(await test_agent_loop())
    else:
        print("\n⚠️  跳过正常执行测试：文档文件不存在")
        print("请先运行 prepare_triviaqa.py 生成测试数据")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    
    # 统计结果
    passed = sum(results)
    total = len(results)
    print(f"通过测试: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败，请检查实现")

if __name__ == "__main__":
    asyncio.run(main())
