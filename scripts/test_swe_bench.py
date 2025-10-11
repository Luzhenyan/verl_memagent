#!/usr/bin/env python3
"""
Test script for SWE-Bench dataset and functionality.
"""

import json
from datasets import load_dataset

def test_swe_bench_dataset():
    """Test loading and exploring SWE-Bench dataset."""
    print("Loading SWE-Bench dataset...")
    
    # Load dataset
    dataset = load_dataset('princeton-nlp/SWE-bench', 'default')
    
    print(f"Dataset splits: {list(dataset.keys())}")
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Test examples: {len(dataset['test'])}")
    print(f"Dev examples: {len(dataset['dev'])}")
    
    # Show first example
    print("\n=== First Training Example ===")
    first_example = dataset['train'][0]
    print(f"Instance ID: {first_example['instance_id']}")
    print(f"Repository: {first_example['repo']}")
    print(f"Base Commit: {first_example['base_commit']}")
    print(f"Problem Statement: {first_example['problem_statement'][:200]}...")
    print(f"Test Patch: {first_example['test_patch'][:200]}...")
    
    # Show dataset schema
    print("\n=== Dataset Schema ===")
    print(dataset['train'].features)
    
    return dataset

def test_swe_bench_functionality():
    """Test SWE-Bench functionality."""
    print("\n=== Testing SWE-Bench Functionality ===")
    
    try:
        from swebench import get_dataset, get_eval_report
        print("SWE-Bench functions imported successfully")
        
        # Test getting dataset
        dataset = get_dataset('default')
        print(f"Dataset loaded via SWE-Bench: {len(dataset)} examples")
        
        # Show a few examples
        for i in range(3):
            example = dataset[i]
            print(f"\nExample {i+1}:")
            print(f"  Instance ID: {example['instance_id']}")
            print(f"  Repository: {example['repo']}")
            print(f"  Problem: {example['problem_statement'][:100]}...")
            
    except Exception as e:
        print(f"Error testing SWE-Bench functionality: {e}")

def test_simple_swe_environment():
    """Test our simple SWE environment."""
    print("\n=== Testing Simple SWE Environment ===")
    
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        from verl.tools.swe_reading_tools import SWEEnvironment
        
        # Create SWE environment
        swe = SWEEnvironment()
        
        # Test file operations
        test_content = "这是一个测试文件。包含一些中文内容。用于测试SWE功能。"
        test_file = "/tmp/swe_test.txt"
        
        # Write file
        success = swe.write_file(test_file, test_content)
        print(f"Write file success: {success}")
        
        # Read file
        content = swe.read_file(test_file)
        print(f"Read file content: {content}")
        
        # Test code execution
        code = """
def process_text(text):
    words = text.split('。')
    return [word.strip() for word in words if word.strip()]

result = process_text(text)
"""
        result = swe.run_code(code, {"text": test_content})
        print(f"Code execution result: {result}")
        
        print("Simple SWE environment test completed successfully!")
        
    except Exception as e:
        print(f"Error testing simple SWE environment: {e}")

def main():
    """Main test function."""
    print("Starting SWE-Bench tests...\n")
    
    # Test dataset loading
    dataset = test_swe_bench_dataset()
    
    # Test SWE-Bench functionality
    test_swe_bench_functionality()
    
    # Test our simple SWE environment
    test_simple_swe_environment()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
