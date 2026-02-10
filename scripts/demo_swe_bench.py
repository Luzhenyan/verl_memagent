#!/usr/bin/env python3
"""
Demo script showing how to use our SWE tools with SWE-Bench data.
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

async def demo_swe_bench_processing():
    """Demo processing SWE-Bench data with our SWE tools."""
    print("=== SWE-Bench Processing Demo ===\n")
    
    # Load SWE-Bench training data
    try:
        with open('/tmp/swe_bench_training_data.json', 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        
        # Use the first example
        task = training_data[0]
        print(f"Processing task: {task['swe_bench_info']['instance_id']}")
        print(f"Repository: {task['swe_bench_info']['repo']}")
        print(f"Question: {task['question']}")
        print(f"Number of segments: {len(task['segments'])}")
        
        # Create SWE tools
        swe = SWEEnvironment()
        
        # Step 1: Read and analyze the document
        print("\n--- Step 1: Reading and Analyzing Document ---")
        content = swe.read_file(task['file_path'])
        print(f"Document loaded: {len(content)} characters")
        
        # Use SWE to analyze the problem
        analysis_code = """
def analyze_software_issue(text):
    lines = text.split('\\n')
    analysis = {
        'issue_type': 'unknown',
        'affected_components': [],
        'severity': 'medium',
        'key_phrases': []
    }
    
    for line in lines:
        line_lower = line.lower()
        if any(word in line_lower for word in ['fail', 'error', 'bug', 'issue']):
            analysis['issue_type'] = 'bug'
        if any(word in line_lower for word in ['cli', 'command', 'hook']):
            analysis['affected_components'].append('CLI')
        if any(word in line_lower for word in ['quiet', 'verbose', 'output']):
            analysis['key_phrases'].append('output control')
    
    return analysis

result = analyze_software_issue(text)
"""
        analysis = swe.run_code(analysis_code, {"text": content})
        print(f"SWE Analysis: {analysis}")
        
        # Step 2: Generate segment summaries
        print("\n--- Step 2: Generating Segment Summaries ---")
        segment_summaries = {}
        
        for i, segment in enumerate(task['segments']):
            summary_code = f"""
def summarize_segment(text):
    lines = text.split('\\n')
    key_points = []
    
    for line in lines:
        if len(line.strip()) > 20:
            if any(keyword in line.lower() for keyword in ['enable', 'quiet', 'verbose', 'cli', 'hook', 'option']):
                key_points.append(line.strip())
    
    return '\\n'.join(key_points[:2]) if key_points else 'Segment contains relevant information'

result = summarize_segment(text)
"""
            summary = swe.run_code(summary_code, {"text": segment})
            segment_summaries[str(i)] = summary
            print(f"Segment {i} summary: {summary[:100]}...")
        
        # Step 3: Update current summary
        print("\n--- Step 3: Updating Current Summary ---")
        update_code = """
def update_summary(segments, question):
    relevant_info = []
    
    for segment in segments.values():
        if any(keyword in segment.lower() for keyword in ['quiet', 'verbose', 'cli', 'hook']):
            relevant_info.append(segment)
    
    if relevant_info:
        return '\\n'.join(relevant_info[:3])
    else:
        return 'No specific information found'

result = update_summary(segments, question)
"""
        current_summary = swe.run_code(update_code, {
            "segments": segment_summaries,
            "question": task['question']
        })
        print(f"Current summary: {current_summary}")
        
        # Step 4: Generate final answer
        print("\n--- Step 4: Generating Final Answer ---")
        answer_code = """
def generate_answer(summary, question):
    if 'quiet' in summary.lower() and 'verbose' in summary.lower():
        return "The issue is that SQLFluff CLI needs a quiet mode option to reduce verbosity for use in pre-commit hooks."
    else:
        return "Based on the analysis, this appears to be a feature request for CLI output control."
    
result = generate_answer(summary, question)
"""
        final_answer = swe.run_code(answer_code, {
            "summary": current_summary,
            "question": task['question']
        })
        print(f"Final answer: {final_answer}")
        
        print("\n✅ SWE-Bench processing demo completed successfully!")
        
    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()

async def demo_tool_integration():
    """Demo using our SWE tools with async interface."""
    print("\n=== SWE Tool Integration Demo ===\n")
    
    try:
        # Create tool schemas
        read_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "swe_read_document",
                "description": "Use SWE to read and segment a document",
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
        
        write_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "swe_write_summary",
                "description": "Use SWE to generate a summary for document content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                    },
                    "required": ["content"],
                }
            }
        })
        
        # Create tools
        read_tool = SWEReadDocumentTool({}, read_schema)
        write_tool = SWEWriteSummaryTool({}, write_schema)
        
        # Test reading
        print("--- Testing Read Tool ---")
        instance_id, response = await read_tool.create()
        parameters = {
            "file_path": "/tmp/swe_bench_sqlfluff__sqlfluff-4764.txt",
            "segment_index": 0
        }
        response, reward, metrics = await read_tool.execute(instance_id, parameters)
        print(f"Read response: {response.text[:100]}...")
        print(f"Reward: {reward}")
        
        # Test writing summary
        print("\n--- Testing Write Summary Tool ---")
        write_instance_id, write_response = await write_tool.create()
        write_parameters = {
            "content": "Enable quiet mode/no-verbose in CLI for use in pre-commit hook. There seems to be only an option to increase the level of verbosity when using SQLFluff CLI, not to limit it further."
        }
        write_response, write_reward, write_metrics = await write_tool.execute(write_instance_id, write_parameters)
        print(f"Write response: {write_response.text}")
        print(f"Reward: {write_reward}")
        
        # Clean up
        await read_tool.release(instance_id)
        await write_tool.release(write_instance_id)
        
        print("\n✅ SWE tool integration demo completed successfully!")
        
    except Exception as e:
        print(f"Error in tool integration demo: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main demo function."""
    print("Starting SWE-Bench Demo...\n")
    
    # Demo 1: SWE-Bench processing
    await demo_swe_bench_processing()
    
    # Demo 2: Tool integration
    await demo_tool_integration()
    
    print("\n🎉 All demos completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
