#!/usr/bin/env python3
"""
准备标准格式的Parquet训练数据
"""

import json
import logging
import pandas as pd
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
    """转换为VERL训练格式"""
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
        
        # 构建prompt（用户消息）
        prompt = [
            {
                "role": "user",
                "content": f"""你是一个智能阅读助手，需要阅读文档并回答问题。

问题：{sample["question"]}

文档内容：
{doc_content.strip()}

请使用工具来帮助回答问题。"""
            }
        ]
        
        # 构建response（助手消息）
        response = [
            {
                "role": "assistant",
                "content": f"基于文档内容，答案是：{sample['answer']}"
            }
        ]
        
        # 创建VERL格式的数据
        verl_sample = {
            "prompt": prompt,
            "response": response,
            "question": sample["question"],
            "answer": sample["answer"],
            "extra_info": {
                "index": i,
                "tools_kwargs": {"dummy": "value"},
                "interaction_kwargs": {"dummy": "value"},
                "need_tools_kwargs": True
            }
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
        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
        
        train_df.to_parquet(f"{output_dir}/train.parquet", index=False)
        val_df.to_parquet(f"{output_dir}/val.parquet", index=False)
        
        logger.info(f"数据准备完成！")
        logger.info(f"训练集: {len(train_data)} 个样本")
        logger.info(f"验证集: {len(val_data)} 个样本")
        logger.info(f"数据保存在: {output_dir}")
        
    except Exception as e:
        logger.error(f"数据准备失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
