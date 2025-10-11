#!/usr/bin/env python3
"""
测试分段阅读工具的功能
验证文件操作工具是否能正常工作
"""

import sys
import os
import json
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from verl.utils.tools.segmented_reading_tools import (
    ReadSegmentFileTool,
    WriteSummaryFileTool,
    ReadSummaryFileTool,
    GetDocumentInfoTool
)

async def test_read_segment_file():
    """测试读取段落文件工具"""
    print("=== 测试 ReadSegmentFileTool ===")
    
    tool = ReadSegmentFileTool()
    instance_id = "test_instance"
    
    # 创建工具实例
    await tool.create(instance_id)
    
    # 测试正常情况
    response, reward, metrics = await tool.execute(
        instance_id, 
        {"segment_index": 0, "file_path": "data/triviaqa_docs/document_0.json"}
    )
    print(f"读取第0段: {response.text[:100]}...")
    print(f"奖励: {reward}, 指标: {metrics}")
    
    # 测试读取第1段
    response, reward, metrics = await tool.execute(
        instance_id, 
        {"segment_index": 1, "file_path": "data/triviaqa_docs/document_0.json"}
    )
    print(f"读取第1段: {response.text[:100]}...")
    print(f"奖励: {reward}, 指标: {metrics}")
    
    # 测试错误情况 - 文件不存在
    response, reward, metrics = await tool.execute(
        instance_id, 
        {"segment_index": 0, "file_path": "data/triviaqa_docs/nonexistent.json"}
    )
    print(f"文件不存在: {response.text}")
    
    # 测试错误情况 - 段落索引超出范围
    response, reward, metrics = await tool.execute(
        instance_id, 
        {"segment_index": 100, "file_path": "data/triviaqa_docs/document_0.json"}
    )
    print(f"段落索引超出范围: {response.text}")
    
    # 释放工具实例
    await tool.release(instance_id)
    print()

async def test_write_summary_file():
    """测试写入总结文件工具"""
    print("=== 测试 WriteSummaryFileTool ===")
    
    tool = WriteSummaryFileTool()
    instance_id = "test_instance"
    
    # 创建工具实例
    await tool.create(instance_id)
    
    # 测试写入总结
    test_summary = "这是测试总结内容，包含重要信息。"
    test_file = "data/triviaqa_docs/test_summary.txt"
    
    response, reward, metrics = await tool.execute(
        instance_id,
        {"summary": test_summary, "file_path": test_file}
    )
    print(f"写入总结: {response.text}")
    print(f"奖励: {reward}, 指标: {metrics}")
    
    # 验证文件是否创建
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"文件内容: {content}")
    else:
        print("文件创建失败")
    
    # 释放工具实例
    await tool.release(instance_id)
    print()

async def test_read_summary_file():
    """测试读取总结文件工具"""
    print("=== 测试 ReadSummaryFileTool ===")
    
    tool = ReadSummaryFileTool()
    instance_id = "test_instance"
    
    # 创建工具实例
    await tool.create(instance_id)
    
    # 测试读取存在的总结文件
    test_file = "data/triviaqa_docs/test_summary.txt"
    if os.path.exists(test_file):
        response, reward, metrics = await tool.execute(
            instance_id,
            {"file_path": test_file}
        )
        print(f"读取总结: {response.text}")
        print(f"奖励: {reward}, 指标: {metrics}")
    else:
        print("测试文件不存在，跳过测试")
    
    # 测试读取不存在的文件
    response, reward, metrics = await tool.execute(
        instance_id,
        {"file_path": "data/triviaqa_docs/nonexistent_summary.txt"}
    )
    print(f"读取不存在的文件: {response.text}")
    
    # 释放工具实例
    await tool.release(instance_id)
    print()

async def test_get_document_info():
    """测试获取文档信息工具"""
    print("=== 测试 GetDocumentInfoTool ===")
    
    tool = GetDocumentInfoTool()
    instance_id = "test_instance"
    
    # 创建工具实例
    await tool.create(instance_id)
    
    # 测试获取文档信息
    response, reward, metrics = await tool.execute(
        instance_id,
        {"file_path": "data/triviaqa_docs/document_0.json"}
    )
    print(f"文档信息: {response.text}")
    print(f"奖励: {reward}, 指标: {metrics}")
    
    # 测试文件不存在的情况
    response, reward, metrics = await tool.execute(
        instance_id,
        {"file_path": "data/triviaqa_docs/nonexistent.json"}
    )
    print(f"文件不存在: {response.text}")
    
    # 释放工具实例
    await tool.release(instance_id)
    print()

async def test_tool_integration():
    """测试工具集成 - 模拟完整的阅读流程"""
    print("=== 测试工具集成 ===")
    
    read_tool = ReadSegmentFileTool()
    write_tool = WriteSummaryFileTool()
    read_summary_tool = ReadSummaryFileTool()
    info_tool = GetDocumentInfoTool()
    
    doc_file = "data/triviaqa_docs/document_0.json"
    summary_file = "data/triviaqa_docs/integration_summary.txt"
    
    # 1. 获取文档信息
    print("1. 获取文档信息:")
    await info_tool.create("info_instance")
    info_response, info_reward, info_metrics = await info_tool.execute(
        "info_instance",
        {"file_path": doc_file}
    )
    print(f"   {info_response.text}")
    
    # 2. 读取第0段
    print("\n2. 读取第0段:")
    await read_tool.create("read_instance")
    segment_response, segment_reward, segment_metrics = await read_tool.execute(
        "read_instance",
        {"segment_index": 0, "file_path": doc_file}
    )
    print(f"   {segment_response.text[:100]}...")
    
    # 3. 写入初始总结
    print("\n3. 写入初始总结:")
    await write_tool.create("write_instance")
    initial_summary = "已读取第1段：关于英格兰的基本信息。"
    write_response, write_reward, write_metrics = await write_tool.execute(
        "write_instance",
        {"summary": initial_summary, "file_path": summary_file}
    )
    print(f"   {write_response.text}")
    
    # 4. 读取第1段
    print("\n4. 读取第1段:")
    segment_response, segment_reward, segment_metrics = await read_tool.execute(
        "read_instance",
        {"segment_index": 1, "file_path": doc_file}
    )
    print(f"   {segment_response.text[:100]}...")
    
    # 5. 更新总结
    print("\n5. 更新总结:")
    updated_summary = "已读取第1-2段：关于英格兰的基本信息和历史背景。"
    write_response, write_reward, write_metrics = await write_tool.execute(
        "write_instance",
        {"summary": updated_summary, "file_path": summary_file}
    )
    print(f"   {write_response.text}")
    
    # 6. 读取当前总结
    print("\n6. 读取当前总结:")
    await read_summary_tool.create("read_summary_instance")
    current_summary_response, current_summary_reward, current_summary_metrics = await read_summary_tool.execute(
        "read_summary_instance",
        {"file_path": summary_file}
    )
    print(f"   {current_summary_response.text}")
    
    # 释放所有工具实例
    await info_tool.release("info_instance")
    await read_tool.release("read_instance")
    await write_tool.release("write_instance")
    await read_summary_tool.release("read_summary_instance")
    
    print()

async def main():
    """主函数"""
    print("开始测试分段阅读工具...")
    print("=" * 50)
    
    # 确保测试目录存在
    os.makedirs("data/triviaqa_docs", exist_ok=True)
    
    # 运行各项测试
    await test_read_segment_file()
    await test_write_summary_file()
    await test_read_summary_file()
    await test_get_document_info()
    await test_tool_integration()
    
    print("=" * 50)
    print("工具测试完成！")
    
    # 清理测试文件
    test_files = [
        "data/triviaqa_docs/test_summary.txt",
        "data/triviaqa_docs/integration_summary.txt"
    ]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"已清理测试文件: {file}")

if __name__ == "__main__":
    asyncio.run(main())
