#!/usr/bin/env python3
"""
调试TriviaQA单个样本的数据结构
"""

from datasets import load_dataset

def main():
    print("加载TriviaQA数据集...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia")
    
    print(f"\n获取第一个样本:")
    sample = ds['train'][0]
    
    print(f"样本类型: {type(sample)}")
    print(f"样本键: {list(sample.keys())}")
    
    print(f"\n=== entity_pages 详细信息 ===")
    entity_pages = sample["entity_pages"]
    print(f"entity_pages类型: {type(entity_pages)}")
    print(f"entity_pages键: {list(entity_pages.keys())}")
    
    for key, value in entity_pages.items():
        print(f"\n{key}:")
        print(f"  类型: {type(value)}")
        if isinstance(value, list):
            print(f"  长度: {len(value)}")
            if value:
                print(f"  第一个元素类型: {type(value[0])}")
                print(f"  第一个元素内容: {str(value[0])[:200]}...")
        else:
            print(f"  值: {str(value)[:200]}...")
    
    print(f"\n=== 测试访问 ===")
    try:
        wiki_contexts = sample["entity_pages"]["wiki_context"]
        print(f"wiki_contexts类型: {type(wiki_contexts)}")
        print(f"wiki_contexts长度: {len(wiki_contexts)}")
        if wiki_contexts:
            print(f"第一个wiki_context: {str(wiki_contexts[0])[:200]}...")
    except Exception as e:
        print(f"访问wiki_context失败: {e}")
        print(f"错误类型: {type(e)}")

if __name__ == "__main__":
    main()






