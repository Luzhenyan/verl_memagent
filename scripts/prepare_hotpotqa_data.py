#!/usr/bin/env python3
"""
准备HotpotQA训练数据
将HotpotQA数据转换为VERL训练所需的格式
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_hotpotqa_data(data_path: str) -> List[Dict[str, Any]]:
    """加载HotpotQA数据"""
    logger.info(f"加载HotpotQA数据: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"加载了 {len(data)} 个样本")
    return data


def create_document_from_context(context: List) -> str:
    """从HotpotQA的context创建文档"""
    doc_content = ""
    for title, sentences in context:
        doc_content += f"【{title}】\n"
        doc_content += "。".join(sentences) + "。\n\n"
    return doc_content.strip()


def convert_to_verl_format(hotpotqa_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将HotpotQA数据转换为VERL训练格式"""
    logger.info("转换数据格式...")
    
    verl_data = []
    
    for i, sample in enumerate(hotpotqa_data):
        if i % 1000 == 0:
            logger.info(f"处理进度: {i}/{len(hotpotqa_data)}")
        
        # 创建文档内容
        doc_content = create_document_from_context(sample['context'])
        
        # 创建VERL格式的数据
        verl_sample = {
            "id": sample.get("_id", f"sample_{i}"),
            "question": sample["question"],
            "answer": sample["answer"],
            "document_content": doc_content,
            "context": sample["context"],
            "supporting_facts": sample["supporting_facts"],
            "type": sample.get("type", "unknown"),
            "level": sample.get("level", "medium")
        }
        
        verl_data.append(verl_sample)
    
    logger.info(f"转换完成，共 {len(verl_data)} 个样本")
    return verl_data


def split_data(data: List[Dict[str, Any]], train_ratio: float = 0.8) -> tuple:
    """分割训练和验证数据"""
    logger.info(f"分割数据，训练比例: {train_ratio}")
    
    total_samples = len(data)
    train_size = int(total_samples * train_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    logger.info(f"训练集: {len(train_data)} 个样本")
    logger.info(f"验证集: {len(val_data)} 个样本")
    
    return train_data, val_data


def save_data(data: List[Dict[str, Any]], output_path: str):
    """保存数据"""
    logger.info(f"保存数据到: {output_path}")
    
    # 确保输出目录存在
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为JSON格式
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"数据保存完成: {output_path}")


def create_sample_data(data: List[Dict[str, Any]], num_samples: int = 10) -> List[Dict[str, Any]]:
    """创建小样本数据用于测试"""
    logger.info(f"创建 {num_samples} 个样本用于测试")
    
    # 选择不同类型的样本
    sample_types = {}
    for sample in data:
        sample_type = sample.get("type", "unknown")
        if sample_type not in sample_types:
            sample_types[sample_type] = []
        sample_types[sample_type].append(sample)
    
    # 从每种类型中选择样本
    selected_samples = []
    samples_per_type = max(1, num_samples // len(sample_types))
    
    for sample_type, samples in sample_types.items():
        selected = samples[:samples_per_type]
        selected_samples.extend(selected)
        logger.info(f"类型 '{sample_type}': {len(selected)} 个样本")
    
    # 如果样本不够，从剩余数据中补充
    if len(selected_samples) < num_samples:
        remaining = [s for s in data if s not in selected_samples]
        additional = remaining[:num_samples - len(selected_samples)]
        selected_samples.extend(additional)
    
    logger.info(f"总共选择了 {len(selected_samples)} 个测试样本")
    return selected_samples


def main():
    """主函数"""
    logger.info("开始准备HotpotQA训练数据")
    
    # 输入和输出路径
    input_path = "/home/luzhenyan/datasets/hotpot_dev_distractor_v1.json"
    output_dir = "/home/luzhenyan/data/segmented_docs"
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. 加载数据
        hotpotqa_data = load_hotpotqa_data(input_path)
        
        # 2. 转换格式
        verl_data = convert_to_verl_format(hotpotqa_data)
        
        # 3. 分割数据
        train_data, val_data = split_data(verl_data, train_ratio=0.8)
        
        # 4. 保存完整数据
        save_data(train_data, f"{output_dir}/train.json")
        save_data(val_data, f"{output_dir}/val.json")
        
        # 5. 创建小样本数据用于测试
        test_samples = create_sample_data(verl_data, num_samples=20)
        save_data(test_samples, f"{output_dir}/test_samples.json")
        
        # 6. 显示数据统计
        logger.info("\n=== 数据统计 ===")
        logger.info(f"总样本数: {len(verl_data)}")
        logger.info(f"训练集: {len(train_data)}")
        logger.info(f"验证集: {len(val_data)}")
        logger.info(f"测试样本: {len(test_samples)}")
        
        # 显示样本类型分布
        type_counts = {}
        for sample in verl_data:
            sample_type = sample.get("type", "unknown")
            type_counts[sample_type] = type_counts.get(sample_type, 0) + 1
        
        logger.info("样本类型分布:")
        for sample_type, count in type_counts.items():
            logger.info(f"  {sample_type}: {count}")
        
        # 显示示例
        logger.info("\n=== 示例数据 ===")
        example = verl_data[0]
        logger.info(f"问题: {example['question']}")
        logger.info(f"答案: {example['answer']}")
        logger.info(f"文档长度: {len(example['document_content'])} 字符")
        logger.info(f"支持事实: {example['supporting_facts']}")
        
        logger.info("\n✅ 数据准备完成！")
        logger.info(f"数据保存在: {output_dir}")
        
    except Exception as e:
        logger.error(f"数据准备失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
