#!/usr/bin/env python3
"""
准备TriviaQA数据集，符合VERL官方数据格式要求
使用mandarjoshi/trivia_qa的rc.wikipedia配置
"""

import logging
import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_triviaqa_data():
    """加载TriviaQA数据集"""
    logger.info("加载TriviaQA数据集...")
    
    try:
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia")
        logger.info(f"数据集加载成功！")
        logger.info(f"训练集: {len(ds['train'])} 个样本")
        logger.info(f"验证集: {len(ds['validation'])} 个样本")
        logger.info(f"测试集: {len(ds['test'])} 个样本")
        return ds
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        raise


def segment_document(context, max_length=2048):
    """将文档按指定长度分段，在句子边界处截断"""
    segments = []
    start = 0
    segment_index = 0
    
    while start < len(context):
        end = min(start + max_length, len(context))
        
        # 如果不是最后一段，尝试在句子边界处截断
        if end < len(context):
            # 寻找最近的句子结束符
            sentence_endings = ['.', '!', '?', '\n\n']
            best_end = end
            
            for ending in sentence_endings:
                # 在目标位置附近寻找句子结束符
                search_start = max(start + max_length - 200, start)
                search_end = min(start + max_length + 200, len(context))
                
                for i in range(search_start, search_end):
                    if context[i] in sentence_endings:
                        # 找到句子结束符，更新结束位置
                        if abs(i - end) < abs(best_end - end):
                            best_end = i + 1
            
            end = best_end
        
        segment_content = context[start:end].strip()
        if segment_content:  # 跳过空内容
            segments.append({
                "title": f"段落{segment_index + 1}",
                "content": segment_content,
                "index": segment_index
            })
            segment_index += 1
        
        start = end
    
    return segments


def save_document_file(sample, segments, output_dir, doc_index):
    """保存文档到JSON文件"""
    doc_file_path = output_dir / f"document_{doc_index}.json"
    
    doc_data = {
        "question": sample["question"],
        "segments": segments,
        "num_segments": len(segments)
    }
    
    with open(doc_file_path, 'w', encoding='utf-8') as f:
        json.dump(doc_data, f, ensure_ascii=False, indent=2)
    
    return str(doc_file_path)


def create_verl_sample(sample, doc_file_path, segments, split="train"):
    """创建符合VERL官方格式的数据样本"""
    # 根据VERL官方文档要求，每个数据项必须包含以下5个字段：
    verl_sample = {
        # 1. data_source: 数据集名称，用于索引对应的奖励函数
        "data_source": "segmented_reading",
        
        # 2. prompt: 使用huggingface chat_template格式构建
        "prompt": [
            {
                "role": "user", 
                "content": f"请阅读文档并回答问题：{sample['question']}\n\n请使用工具开始阅读文档。"
            }
        ],
        
        # 3. ability: 定义任务类别
        "ability": "reading_comprehension",
        
        # 4. reward_model: 包含ground_truth字段，用于评估
        "reward_model": {
            "style": "rule",
            "ground_truth": sample['answer']['value']
        },
        
        # 5. extra_info: 记录当前prompt的一些信息
        "extra_info": {
            "question": sample["question"],
            "document_file": doc_file_path,
            "num_segments": len(segments),
            "split": split
        },
        
        # 6. agent_name: 指定使用哪个Agent Loop（关键字段！）
        "agent_name": "segmented_reading_agent"
    }
    
    return verl_sample


def main():
    """主函数"""
    logger.info("开始准备TriviaQA训练数据")
    
    # 输出路径
    output_dir = Path("/home/luzhenyan/data/triviaqa_docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. 加载数据
        dataset = load_triviaqa_data()
        
        # 2. 处理训练集（限制样本数量）
        logger.info("处理训练集...")
        train_samples = dataset['train'].select(range(20))  # 只取20个样本
        train_verl_samples = []
        
        for i, sample in enumerate(train_samples):
            logger.info(f"处理训练样本 {i+1}/20")
            
            # 获取文档内容
            wiki_contexts = sample["entity_pages"]["wiki_context"]
            if wiki_contexts:
                # 合并所有文档内容
                combined_context = "\n\n".join(wiki_contexts)
                
                # 分段
                segments = segment_document(combined_context)
                
                # 保存文档文件
                doc_file_path = save_document_file(sample, segments, output_dir, i)
                
                # 创建VERL格式数据
                verl_sample = create_verl_sample(sample, doc_file_path, segments, "train")
                train_verl_samples.append(verl_sample)
            else:
                logger.warning(f"样本 {i} 没有文档内容，跳过")
        
        # 3. 处理验证集（限制样本数量）
        logger.info("处理验证集...")
        val_samples = dataset['validation'].select(range(4))  # 只取4个样本
        val_verl_samples = []
        
        for i, sample in enumerate(val_samples):
            logger.info(f"处理验证样本 {i+1}/4")
            
            # 获取文档内容
            wiki_contexts = sample["entity_pages"]["wiki_context"]
            if wiki_contexts:
                # 合并所有文档内容
                combined_context = "\n\n".join(wiki_contexts)
                
                # 分段
                segments = segment_document(combined_context)
                
                # 保存文档文件
                doc_file_path = save_document_file(sample, segments, output_dir, len(train_samples) + i)
                
                # 创建VERL格式数据
                verl_sample = create_verl_sample(sample, doc_file_path, segments, "validation")
                val_verl_samples.append(verl_sample)
            else:
                logger.warning(f"验证样本 {i} 没有文档内容，跳过")
        
        # 4. 保存训练数据
        train_df = pd.DataFrame(train_verl_samples)
        train_path = output_dir / "train_small.parquet"
        train_df.to_parquet(train_path, index=False)
        logger.info(f"训练数据保存到: {train_path}")
        
        # 5. 保存验证数据
        val_df = pd.DataFrame(val_verl_samples)
        val_path = output_dir / "val.parquet"
        val_df.to_parquet(val_path, index=False)
        logger.info(f"验证数据保存到: {val_path}")
        
        # 6. 显示示例
        logger.info("数据格式示例:")
        logger.info(f"训练样本数量: {len(train_verl_samples)}")
        logger.info(f"验证样本数量: {len(val_verl_samples)}")
        logger.info(f"文档文件数量: {len(train_samples) + len(val_samples)}")
        
        if train_verl_samples:
            logger.info("第一个训练样本结构:")
            for key, value in train_verl_samples[0].items():
                if key == "prompt":
                    logger.info(f"  {key}: {len(value)} 条消息")
                elif key == "extra_info":
                    logger.info(f"  {key}: {list(value.keys())}")
                else:
                    logger.info(f"  {key}: {value}")
        
        logger.info("数据准备完成！")
        
    except Exception as e:
        logger.error(f"数据准备失败: {e}")
        raise


if __name__ == "__main__":
    main()
