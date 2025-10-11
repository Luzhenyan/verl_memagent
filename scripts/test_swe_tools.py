#!/usr/bin/env python3
"""
Test script for SWE reading tools.
"""

import asyncio
import json
import sys
import os

# Add verl to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from verl.tools.swe_reading_tools import (
    SWEEnvironment,
    SWEReadDocumentTool,
    SWEWriteSummaryTool,
    SWEUpdateSummaryTool,
    SWEGenerateAnswerTool,
)
from verl.tools.schemas import OpenAIFunctionToolSchema

async def test_swe_environment():
    """Test SWEEnvironment."""
    print("Testing SWEEnvironment...")
    
    swe = SWEEnvironment()
    
    # Test file operations
    test_content = "这是一个测试文档。包含多个句子。用于测试SWE功能。"
    test_file = "/tmp/swe_test.txt"
    
    # Write file
    success = swe.write_file(test_file, test_content)
    print(f"Write file success: {success}")
    
    # Read file
    content = swe.read_file(test_file)
    print(f"Read file content: {content}")
    
    # Test code execution
    code = """
def add_numbers(a, b):
    return a + b

result = add_numbers(5, 3)
"""
    result = swe.run_code(code, {"a": 5, "b": 3})
    print(f"Code execution result: {result}")
    
    print("SWEEnvironment test completed!\n")

async def test_swe_read_document_tool():
    """Test SWEReadDocumentTool."""
    print("Testing SWEReadDocumentTool...")
    
    # Create tool
    tool_schema = OpenAIFunctionToolSchema.model_validate({
        "type": "function",
        "function": {
            "name": "swe_read_document",
            "description": "Use SWE to read and segment a document",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "segment_index": {"type": "integer"},
                    "segment_method": {"type": "string"},
                },
                "required": ["file_path", "segment_index"],
            }
        }
    })
    
    tool = SWEReadDocumentTool({}, tool_schema)
    
    # Create instance
    instance_id, response = await tool.create()
    print(f"Created instance: {instance_id}")
    
    # Test reading segment with semantic segmentation
    parameters = {
        "file_path": "/user/luzhenyan/data/segmented_docs/doc1.txt",
        "segment_index": 0,
        "segment_method": "semantic"
    }
    
    response, reward, metrics = await tool.execute(instance_id, parameters)
    print(f"Response: {response.text}")
    print(f"Reward: {reward}")
    
    # Clean up
    await tool.release(instance_id)
    print("SWEReadDocumentTool test completed!\n")

async def test_swe_write_summary_tool():
    """Test SWEWriteSummaryTool."""
    print("Testing SWEWriteSummaryTool...")
    
    # Create tool
    tool_schema = OpenAIFunctionToolSchema.model_validate({
        "type": "function",
        "function": {
            "name": "swe_write_summary",
            "description": "Use SWE to generate a summary for document content",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "summary_style": {"type": "string"},
                },
                "required": ["content"],
            }
        }
    })
    
    tool = SWEWriteSummaryTool({}, tool_schema)
    
    # Create instance
    instance_id, response = await tool.create()
    print(f"Created instance: {instance_id}")
    
    # Test writing summary
    parameters = {
        "content": "人工智能的概念最早由艾伦·图灵在1950年提出。他提出了著名的图灵测试，用于判断机器是否具有智能。在20世纪60年代，人工智能研究主要集中在符号推理和专家系统上。",
        "summary_style": "concise"
    }
    
    response, reward, metrics = await tool.execute(instance_id, parameters)
    print(f"Response: {response.text}")
    print(f"Reward: {reward}")
    
    # Clean up
    await tool.release(instance_id)
    print("SWEWriteSummaryTool test completed!\n")

async def test_swe_update_summary_tool():
    """Test SWEUpdateSummaryTool."""
    print("Testing SWEUpdateSummaryTool...")
    
    # Create tool
    tool_schema = OpenAIFunctionToolSchema.model_validate({
        "type": "function",
        "function": {
            "name": "swe_update_summary",
            "description": "Use SWE to update comprehensive summary based on segment summaries",
            "parameters": {
                "type": "object",
                "properties": {
                    "segment_summaries": {"type": "string"},
                    "question": {"type": "string"},
                    "current_summary": {"type": "string"},
                },
                "required": ["segment_summaries", "question", "current_summary"],
            }
        }
    })
    
    tool = SWEUpdateSummaryTool({}, tool_schema)
    
    # Create instance
    instance_id, response = await tool.create()
    print(f"Created instance: {instance_id}")
    
    # Test updating summary
    segment_summaries = {
        "0": "图灵提出AI概念和图灵测试",
        "1": "60年代发展符号推理和专家系统",
        "2": "80年代兴起机器学习和神经网络"
    }
    
    parameters = {
        "segment_summaries": json.dumps(segment_summaries),
        "question": "人工智能的发展历程是怎样的？",
        "current_summary": "AI从图灵概念提出开始，经历了符号推理的发展"
    }
    
    response, reward, metrics = await tool.execute(instance_id, parameters)
    print(f"Response: {response.text}")
    print(f"Reward: {reward}")
    
    # Clean up
    await tool.release(instance_id)
    print("SWEUpdateSummaryTool test completed!\n")

async def test_swe_generate_answer_tool():
    """Test SWEGenerateAnswerTool."""
    print("Testing SWEGenerateAnswerTool...")
    
    # Create tool
    tool_schema = OpenAIFunctionToolSchema.model_validate({
        "type": "function",
        "function": {
            "name": "swe_generate_answer",
            "description": "Use SWE to generate final answer based on summary",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "question": {"type": "string"},
                    "final_answer": {"type": "string"},
                },
                "required": ["summary", "question", "final_answer"],
            }
        }
    })
    
    tool = SWEGenerateAnswerTool({}, tool_schema)
    
    # Create instance
    instance_id, response = await tool.create()
    print(f"Created instance: {instance_id}")
    
    # Test generating answer
    parameters = {
        "summary": "AI从图灵概念提出开始，经历了符号推理、专家系统、机器学习到深度学习的发展历程",
        "question": "人工智能的发展历程是怎样的？",
        "final_answer": "人工智能从1950年图灵提出概念开始，经历了符号推理、专家系统、机器学习、深度学习等发展阶段。"
    }
    
    response, reward, metrics = await tool.execute(instance_id, parameters)
    print(f"Response: {response.text}")
    print(f"Reward: {reward}")
    
    # Clean up
    await tool.release(instance_id)
    print("SWEGenerateAnswerTool test completed!\n")

async def main():
    """Main test function."""
    print("Starting SWE tools tests...\n")
    
    # Test SWE environment
    await test_swe_environment()
    
    # Test all SWE tools
    await test_swe_read_document_tool()
    await test_swe_write_summary_tool()
    await test_swe_update_summary_tool()
    await test_swe_generate_answer_tool()
    
    print("All SWE tools tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
