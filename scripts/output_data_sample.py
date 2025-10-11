#!/usr/bin/env python3
"""
输出数据样本到文件
"""

import pandas as pd
import json

def main():
    # 读取数据
    df = pd.read_parquet('/home/luzhenyan/data/triviaqa_docs/train_small.parquet')
    
    # 获取第一个样本
    sample = df.iloc[0]
    
    # 准备输出数据
    output_data = {
        "question": sample['question'],
        "answer": sample['answer'],
        "context_length": len(sample['context']),
        "num_segments": sample['num_segments'],
        "segments": []
    }
    
    # 添加前3个段落的信息
    for i, segment in enumerate(sample['segments'][:3]):
        output_data['segments'].append({
            "title": segment['title'],
            "index": segment['index'],
            "content_length": len(segment['content']),
            "content_full": segment['content']
        })
    
    # 写入文件
    with open('/home/luzhenyan/verl/datasample.txt', 'w', encoding='utf-8') as f:
        f.write("=== TriviaQA Data Sample ===\n\n")
        f.write(f"Question: {output_data['question']}\n")
        f.write(f"Answer: {output_data['answer']}\n")
        f.write(f"Context Length: {output_data['context_length']} characters\n")
        f.write(f"Number of Segments: {output_data['num_segments']}\n\n")
        
        for i, segment in enumerate(output_data['segments']):
            f.write(f"=== Segment {i+1} ===\n")
            f.write(f"Title: {segment['title']}\n")
            f.write(f"Index: {segment['index']}\n")
            f.write(f"Length: {segment['content_length']} characters\n")
            f.write(f"Content:\n{segment['content_full']}\n\n")
    
    print("Data sample written to /home/luzhenyan/verl/datasample.txt")

if __name__ == "__main__":
    main()
