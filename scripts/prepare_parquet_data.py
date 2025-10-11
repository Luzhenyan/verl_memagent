#!/usr/bin/env python3
"""
准备标准格式的Parquet训练数据
生成VERL训练所需的prompt和response格式
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
    """转换为VERL训练格式"""
    logger.info("转换数据格式...")
    
    verl_data = []
    
    for i, sample in enumerate(hotpotqa_data):
        if i % 1000 == 0:
            logger.info(f"处理进度: {i}/{len(hotpotqa_data)}")
        
        # 创建文档内容
        doc_content = create_document_from_context(sample['context'])
        
        # 构建prompt（用户消息）
        prompt = [
            {
                "role": "user",
                "content": f"""你是一个智能阅读助手，需要阅读文档并回答问题。

问题：{sample["question"]}

文档内容：
{doc_content}

请使用以下工具来帮助回答问题：
1. read_document_segment - 读取文档段落
2. write_segment_summary - 为段落写总结
3. update_current_summary - 更新当前总结
4. generate_final_answer - 生成最终答案

请开始阅读文档并回答问题。"""
            }
        ]
        
        # 构建response（助手消息，包含工具调用）
        response = [
            {
                "role": "assistant",
                "content": "我将开始阅读文档并回答问题。",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_document_segment",
                            "arguments": json.dumps({"segment_index": 0})
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": f"已读取段落0：{sample['context'][0][0] if sample['context'] else '无内容'}"
            },
            {
                "role": "assistant",
                "content": "基于阅读的内容，我的答案是：",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "generate_final_answer",
                            "arguments": json.dumps({"answer": sample["answer"]})
                        }
                    }
                ]
            }
        ]
        
        # 创建VERL格式的数据
        verl_sample = {
            "prompt": prompt,
            "response": response,
            "question": sample["question"],
            "answer": sample["answer"],
            "document_content": doc_content,
            "supporting_facts": sample["supporting_facts"],
            "type": sample.get("type", "unknown"),
            "level": sample.get("level", "medium"),
            "extra_info": {
                "index": i,
                "tools_kwargs": {},
                "interaction_kwargs": {},
                "need_tools_kwargs": True
            }
        }
        
        verl_data.append(verl_sample)
    
    logger.info(f"转换完成，共 {len(verl_data)} 个样本")
    return verl_data


def save_as_parquet(data: List[Dict[str, Any]], output_path: str):
    """保存为Parquet格式"""
    logger.info(f"保存数据到: {output_path}")
    
    # 确保输出目录存在
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 保存为Parquet
    df.to_parquet(output_path, index=False)
    
    logger.info(f"数据保存完成: {output_path}")


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
        
        # 3. 分割数据（80%训练，20%验证）
        train_size = int(len(verl_data) * 0.8)
        train_data = verl_data[:train_size]
        val_data = verl_data[train_size:]
        
        # 4. 保存数据
        save_as_parquet(train_data, f"{output_dir}/train.parquet")
        save_as_parquet(val_data, f"{output_dir}/val.parquet")
        
        # 5. 创建小样本数据用于测试
        test_samples = verl_data[:20]
        save_as_parquet(test_samples, f"{output_dir}/test_samples.parquet")
        
        logger.info(f"数据准备完成！")
        logger.info(f"训练集: {len(train_data)} 个样本")
        logger.info(f"验证集: {len(val_data)} 个样本")
        logger.info(f"测试样本: {len(test_samples)} 个样本")
        logger.info(f"数据保存在: {output_dir}")
        
        # 显示示例
        logger.info("\n=== 示例数据 ===")
        example = verl_data[0]
        logger.info(f"问题: {example['question']}")
        logger.info(f"答案: {example['answer']}")
        logger.info(f"Prompt长度: {len(str(example['prompt']))} 字符")
        logger.info(f"Response长度: {len(str(example['response']))} 字符")
        
    except Exception as e:
        logger.error(f"数据准备失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
