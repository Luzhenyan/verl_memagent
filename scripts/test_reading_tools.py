#!/usr/bin/env python3
"""
Test script for reading tools.
"""

import asyncio
import json
import sys
import os

# Add verl to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from verl.tools.reading_tools import (
    ReadDocumentTool,
    WriteSummaryTool,
    UpdateCurrentSummaryTool,
    GenerateFinalAnswerTool,
)
from verl.tools.schemas import OpenAIFunctionToolSchema

async def test_read_document_tool():
    """Test ReadDocumentTool."""
    print("Testing ReadDocumentTool...")
    
    # Create tool
    tool_schema = OpenAIFunctionToolSchema.model_validate({
        "type": "function",
        "function": {
            "name": "read_document_segment",
            "description": "Read a specific segment of a document",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "segment_index": {"type": "integer"},
                },
                "required": ["file_path", "segment_index"],
            }
        }
    })
    
    tool = ReadDocumentTool({}, tool_schema)
    
    # Create instance
    instance_id, response = await tool.create()
    print(f"Created instance: {instance_id}")
    
    # Test reading segment
    parameters = {
        "file_path": "/user/luzhenyan/data/segmented_docs/doc1.txt",
        "segment_index": 0
    }
    
    response, reward, metrics = await tool.execute(instance_id, parameters)
    print(f"Response: {response.text}")
    print(f"Reward: {reward}")
    
    # Clean up
    await tool.release(instance_id)
    print("ReadDocumentTool test completed!\n")

async def test_write_summary_tool():
    """Test WriteSummaryTool."""
    print("Testing WriteSummaryTool...")
    
    # Create tool
    tool_schema = OpenAIFunctionToolSchema.model_validate({
        "type": "function",
        "function": {
            "name": "write_segment_summary",
            "description": "Write a summary for a document segment",
            "parameters": {
                "type": "object",
                "properties": {
                    "segment_content": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["segment_content", "summary"],
            }
        }
    })
    
    tool = WriteSummaryTool({}, tool_schema)
    
    # Create instance
    instance_id, response = await tool.create()
    print(f"Created instance: {instance_id}")
    
    # Test writing summary
    parameters = {
        "segment_content": "人工智能的概念最早由艾伦·图灵在1950年提出。他提出了著名的图灵测试，用于判断机器是否具有智能。",
        "summary": "图灵提出AI概念和图灵测试"
    }
    
    response, reward, metrics = await tool.execute(instance_id, parameters)
    print(f"Response: {response.text}")
    print(f"Reward: {reward}")
    
    # Clean up
    await tool.release(instance_id)
    print("WriteSummaryTool test completed!\n")

async def test_update_current_summary_tool():
    """Test UpdateCurrentSummaryTool."""
    print("Testing UpdateCurrentSummaryTool...")
    
    # Create tool
    tool_schema = OpenAIFunctionToolSchema.model_validate({
        "type": "function",
        "function": {
            "name": "update_current_summary",
            "description": "Update the current summary based on all segment summaries",
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
    
    tool = UpdateCurrentSummaryTool({}, tool_schema)
    
    # Create instance
    instance_id, response = await tool.create()
    print(f"Created instance: {instance_id}")
    
    # Test updating summary
    segment_summaries = {
        "0": "图灵提出AI概念和图灵测试",
        "1": "60年代发展符号推理和专家系统"
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
    print("UpdateCurrentSummaryTool test completed!\n")

async def test_generate_final_answer_tool():
    """Test GenerateFinalAnswerTool."""
    print("Testing GenerateFinalAnswerTool...")
    
    # Create tool
    tool_schema = OpenAIFunctionToolSchema.model_validate({
        "type": "function",
        "function": {
            "name": "generate_final_answer",
            "description": "Generate final answer based on current summary",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_summary": {"type": "string"},
                    "question": {"type": "string"},
                    "final_answer": {"type": "string"},
                },
                "required": ["current_summary", "question", "final_answer"],
            }
        }
    })
    
    tool = GenerateFinalAnswerTool({}, tool_schema)
    
    # Create instance
    instance_id, response = await tool.create()
    print(f"Created instance: {instance_id}")
    
    # Test generating answer
    parameters = {
        "current_summary": "AI从图灵概念提出开始，经历了符号推理、专家系统、机器学习到深度学习的发展历程",
        "question": "人工智能的发展历程是怎样的？",
        "final_answer": "人工智能从1950年图灵提出概念开始，经历了符号推理、专家系统、机器学习、深度学习等发展阶段。"
    }
    
    response, reward, metrics = await tool.execute(instance_id, parameters)
    print(f"Response: {response.text}")
    print(f"Reward: {reward}")
    
    # Clean up
    await tool.release(instance_id)
    print("GenerateFinalAnswerTool test completed!\n")

async def main():
    """Main test function."""
    print("Starting reading tools tests...\n")
    
    # Test all tools
    await test_read_document_tool()
    await test_write_summary_tool()
    await test_update_current_summary_tool()
    await test_generate_final_answer_tool()
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
