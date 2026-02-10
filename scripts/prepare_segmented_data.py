#!/usr/bin/env python3
"""
Prepare segmented reading training data.
"""

import json
import pandas as pd
import os
from pathlib import Path

def create_sample_data():
    """Create sample training data for segmented reading."""
    
    # Sample documents and questions
    sample_data = [
        {
            "file_path": "/user/wangyicheng/data/segmented_docs/doc1.txt",
            "question": "人工智能的发展历程是怎样的？",
            "segments": [
                "人工智能的概念最早由艾伦·图灵在1950年提出。他提出了著名的图灵测试，用于判断机器是否具有智能。",
                "在20世纪60年代，人工智能研究主要集中在符号推理和专家系统上。这一时期出现了许多重要的AI程序。",
                "到了80年代，机器学习开始兴起。神经网络和深度学习的概念逐渐发展起来。",
                "21世纪以来，随着大数据和计算能力的提升，深度学习取得了突破性进展。"
            ],
            "segment_summaries": [
                "图灵提出AI概念和图灵测试",
                "60年代发展符号推理和专家系统",
                "80年代兴起机器学习和神经网络",
                "21世纪深度学习取得突破"
            ],
            "current_summary": "AI从图灵概念提出开始，经历了符号推理、专家系统、机器学习到深度学习的发展历程",
            "expected_answer": "人工智能从1950年图灵提出概念开始，经历了符号推理、专家系统、机器学习、深度学习等发展阶段。",
            "relevant_segments": [0, 1, 2, 3],
            "difficulty": "medium"
        },
        {
            "file_path": "/user/wangyicheng/data/segmented_docs/doc2.txt",
            "question": "机器学习的主要类型有哪些？",
            "segments": [
                "机器学习主要分为监督学习、无监督学习和强化学习三大类。",
                "监督学习使用标记的训练数据来学习输入和输出之间的映射关系。常见的算法包括线性回归、逻辑回归、支持向量机等。",
                "无监督学习从未标记的数据中发现隐藏的模式和结构。聚类、降维、关联规则挖掘都是无监督学习的典型应用。",
                "强化学习通过与环境交互来学习最优策略。智能体通过试错来最大化累积奖励。"
            ],
            "segment_summaries": [
                "机器学习分为三大类",
                "监督学习使用标记数据",
                "无监督学习发现隐藏模式",
                "强化学习通过交互学习"
            ],
            "current_summary": "机器学习包括监督学习（使用标记数据）、无监督学习（发现隐藏模式）和强化学习（通过交互学习）三大类型",
            "expected_answer": "机器学习主要分为监督学习、无监督学习和强化学习三大类。",
            "relevant_segments": [0, 1, 2, 3],
            "difficulty": "easy"
        },
        {
            "file_path": "/user/wangyicheng/data/segmented_docs/doc3.txt",
            "question": "深度学习在哪些领域取得了突破？",
            "segments": [
                "深度学习在计算机视觉领域取得了显著突破。卷积神经网络在图像分类、目标检测、图像分割等任务上表现优异。",
                "在自然语言处理领域，Transformer架构和预训练语言模型如BERT、GPT系列彻底改变了NLP的发展方向。",
                "语音识别和语音合成技术也因深度学习而大幅提升。端到端的语音识别系统准确率已经接近人类水平。",
                "在医疗诊断、自动驾驶、推荐系统等领域，深度学习也展现出了巨大的应用潜力。"
            ],
            "segment_summaries": [
                "计算机视觉领域突破",
                "自然语言处理领域突破",
                "语音技术大幅提升",
                "多领域应用潜力"
            ],
            "current_summary": "深度学习在计算机视觉、自然语言处理、语音技术以及医疗、自动驾驶等多个领域取得了突破性进展",
            "expected_answer": "深度学习在计算机视觉、自然语言处理、语音识别、医疗诊断、自动驾驶、推荐系统等多个领域取得了突破。",
            "relevant_segments": [0, 1, 2, 3],
            "difficulty": "medium"
        }
    ]
    
    return sample_data

def create_document_files(data_dir: str, sample_data: list):
    """Create document files from sample data."""
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    for i, item in enumerate(sample_data):
        doc_path = data_path / f"doc{i+1}.txt"
        content = "。".join(item["segments"]) + "。"
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Update file path in data
        item["file_path"] = str(doc_path)

def save_parquet_data(data: list, output_path: str):
    """Save data as parquet file."""
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save as parquet
    df.to_parquet(output_path, index=False)
    print(f"Data saved to {output_path}")

def main():
    """Main function to prepare training data."""
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Create document files
    data_dir = "/user/wangyicheng/data/segmented_docs"
    create_document_files(data_dir, sample_data)
    
    # Save training data
    train_path = os.path.join(data_dir, "train.parquet")
    save_parquet_data(sample_data, train_path)
    
    # Create validation data (same as training for now)
    val_path = os.path.join(data_dir, "val.parquet")
    save_parquet_data(sample_data, val_path)
    
    print("Data preparation completed!")
    print(f"Training data: {train_path}")
    print(f"Validation data: {val_path}")

if __name__ == "__main__":
    main()
