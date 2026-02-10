#!/usr/bin/env python3
"""
准备HotpotQA训练数据
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_hotpotqa_data(data_path: str):
    """加载HotpotQA数据"""
    logger.info(f"加载HotpotQA数据: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"加载了 {len(data)} 个样本")
    return data


def convert_to_verl_format(hotpotqa_data):
    """转换为VERL格式"""
    logger.info("转换数据格式...")
    
    verl_data = []
    
    for i, sample in enumerate(hotpotqa_data):
        if i % 1000 == 0:
            logger.info(f"处理进度: {i}/{len(hotpotqa_data)}")
        
        # 创建文档内容
        doc_content = ""
        for title, sentences in sample['context']:
            doc_content += f"【{title}】\n"
            doc_content += "。".join(sentences) + "。\n\n"
        
        # 创建VERL格式的数据
        verl_sample = {
            "id": sample.get("_id", f"sample_{i}"),
            "question": sample["question"],
            "answer": sample["answer"],
            "document_content": doc_content.strip(),
            "context": sample["context"],
            "supporting_facts": sample["supporting_facts"],
            "type": sample.get("type", "unknown"),
            "level": sample.get("level", "medium")
        }
        
        verl_data.append(verl_sample)
    
    logger.info(f"转换完成，共 {len(verl_data)} 个样本")
    return verl_data


def main():
    """主函数"""
    logger.info("开始准备HotpotQA训练数据")
    
    # 输入和输出路径
    input_path = "/home/wangyicheng/datasets/hotpot_dev_distractor_v1.json"
    output_dir = "/home/wangyicheng/data/segmented_docs"
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. 加载数据
        hotpotqa_data = load_hotpotqa_data(input_path)
        
        # 2. 转换格式
        verl_data = convert_to_verl_format(hotpotqa_data)
        
        # 3. 分割数据（80%训练，20%验证）
        train_size = int(len(verl_data) * 0.8)
        train_data = verl_data[:train_size]
        val_data = verl_data[train_size:]
        
        # 4. 保存数据
        with open(f"{output_dir}/train.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(f"{output_dir}/val.json", 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        # 5. 创建小样本数据用于测试
        test_samples = verl_data[:20]
        with open(f"{output_dir}/test_samples.json", 'w', encoding='utf-8') as f:
            json.dump(test_samples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据准备完成！")
        logger.info(f"训练集: {len(train_data)} 个样本")
        logger.info(f"验证集: {len(val_data)} 个样本")
        logger.info(f"测试样本: {len(test_samples)} 个样本")
        logger.info(f"数据保存在: {output_dir}")
        
    except Exception as e:
        logger.error(f"数据准备失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
