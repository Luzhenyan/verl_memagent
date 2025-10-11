#!/usr/bin/env python3
"""
调试TriviaQA数据集索引问题
"""

from datasets import load_dataset

def main():
    print("加载TriviaQA数据集...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia")
    
    print(f"\n数据集信息:")
    print(f"训练集类型: {type(ds['train'])}")
    print(f"训练集长度: {len(ds['train'])}")
    
    print(f"\n=== 检查前几个样本 ===")
    for i in range(5):
        try:
            sample = ds['train'][i]
            print(f"\n样本 {i}:")
            print(f"  类型: {type(sample)}")
            if isinstance(sample, dict):
                print(f"  键: {list(sample.keys())}")
                if "question" in sample:
                    print(f"  问题: {str(sample['question'])[:100]}...")
            else:
                print(f"  值: {str(sample)[:100]}...")
        except Exception as e:
            print(f"  获取样本 {i} 失败: {e}")
    
    print(f"\n=== 检查数据集切片 ===")
    try:
        subset = ds['train'][:10]
        print(f"切片类型: {type(subset)}")
        print(f"切片长度: {len(subset)}")
        if hasattr(subset, '__getitem__'):
            first_item = subset[0]
            print(f"第一个元素类型: {type(first_item)}")
            if isinstance(first_item, dict):
                print(f"第一个元素键: {list(first_item.keys())}")
    except Exception as e:
        print(f"切片操作失败: {e}")

if __name__ == "__main__":
    main()






