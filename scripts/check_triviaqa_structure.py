#!/usr/bin/env python3
"""
检查TriviaQA数据集结构
"""

from datasets import load_dataset

def main():
    print("加载TriviaQA数据集...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia")
    
    print(f"\n数据集结构:")
    print(f"训练集: {len(ds['train'])} 个样本")
    print(f"验证集: {len(ds['validation'])} 个样本")
    print(f"测试集: {len(ds['test'])} 个样本")
    
    print(f"\n训练集第一个样本:")
    sample = ds['train'][0]
    print(f"类型: {type(sample)}")
    print(f"键: {list(sample.keys())}")
    
    for key, value in sample.items():
        print(f"\n{key}:")
        print(f"  类型: {type(value)}")
        if isinstance(value, dict):
            print(f"  键: {list(value.keys())}")
            for k, v in value.items():
                print(f"    {k}: {type(v)} - {str(v)[:100]}...")
        elif isinstance(value, list):
            print(f"  长度: {len(value)}")
            if value:
                print(f"  第一个元素: {type(value[0])} - {str(value[0])[:100]}...")
        else:
            print(f"  值: {str(value)[:100]}...")

if __name__ == "__main__":
    main()






