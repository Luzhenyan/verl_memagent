"""
SegmentedReadingDataset - 自定义数据集类
集成SegmentedReadingEnvironment到VERL训练流程
"""

import json
import logging
import random
from typing import Dict, List, Any, Optional

import torch
from torch.utils.data import Dataset

from verl.environments.segmented_reading_env import SegmentedReadingEnvironment, SegmentedReadingState

logger = logging.getLogger(__name__)


class SegmentedReadingDataset(Dataset):
    """分段阅读数据集类，集成SegmentedReadingEnvironment"""
    
    def __init__(self, data_files: List[str], tokenizer, processor, config: Dict[str, Any]):
        """
        初始化数据集
        
        Args:
            data_files: 数据文件路径列表
            tokenizer: 分词器
            processor: 处理器
            config: 配置字典
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        
        # 初始化环境
        self.environment = SegmentedReadingEnvironment(config.get("environment", {}))
        
        # 加载数据
        self.data = []
        for data_file in data_files:
            logger.info(f"Loading data from {data_file}")
            self.environment.load_data(data_file)
            self.data.extend(self.environment.hotpotqa_data)
        
        logger.info(f"Loaded {len(self.data)} samples")
        
        # 训练参数
        self.max_prompt_length = config.get("max_prompt_length", 2048)
        self.max_response_length = config.get("max_response_length", 1024)
        self.shuffle = config.get("shuffle", True)
        
        # 工具调用配置
        self.tool_config = config.get("tools", {})
        
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含prompt、response等字段的字典
        """
        # 重置环境到指定样本
        self.environment.reset(episode_id=idx)
        state = self.environment.current_state
        
        # 构建初始prompt
        prompt = self._build_initial_prompt(state)
        
        # 构建工具调用格式的response
        response = self._build_tool_calling_response(state)
        
        # 编码
        encoded_prompt = self.tokenizer(
            prompt,
            max_length=self.max_prompt_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        
        encoded_response = self.tokenizer(
            response,
            max_length=self.max_response_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        
        return {
            "prompt": prompt,
            "response": response,
            "prompt_input_ids": encoded_prompt["input_ids"].squeeze(0),
            "prompt_attention_mask": encoded_prompt["attention_mask"].squeeze(0),
            "response_input_ids": encoded_response["input_ids"].squeeze(0),
            "response_attention_mask": encoded_response["attention_mask"].squeeze(0),
            "sample_id": idx,
            "question": state.question,
            "answer": state.answer,
            "supporting_facts": state.supporting_facts
        }
    
    def _build_initial_prompt(self, state: SegmentedReadingState) -> str:
        """构建初始prompt"""
        prompt = f"""你是一个智能阅读助手，需要阅读文档并回答问题。

问题：{state.question}

可用工具：
1. read_document_segment - 读取文档段落
2. write_segment_summary - 为段落写总结
3. update_current_summary - 更新当前总结
4. generate_final_answer - 生成最终答案

请开始阅读文档并回答问题。你可以：
1. 先读取一些段落
2. 为每个段落写总结
3. 基于总结更新当前理解
4. 重复上述步骤直到能够回答问题
5. 生成最终答案

开始行动："""
        
        return prompt
    
    def _build_tool_calling_response(self, state: SegmentedReadingState) -> str:
        """构建工具调用格式的response"""
        # 模拟一个合理的工具调用序列
        response_parts = []
        
        # 1. 读取前几个段落
        num_segments_to_read = min(3, len(state.context))
        for i in range(num_segments_to_read):
            title, sentences = state.context[i]
            segment_content = f"【{title}】\n" + "。".join(sentences) + "。"
            
            # 工具调用：读取段落
            response_parts.append(f"""<tool_call>
<invoke name="read_document_segment">
<parameter name="segment_index">{i}</parameter>
</invoke>
</tool_call>

<tool_result>
<result name="read_document_segment">
<parameter name="segment_content">{segment_content}</parameter>
<parameter name="segment_index">{i}</parameter>
<parameter name="title">{title}</parameter>
</result>
</tool_result>""")
            
            # 工具调用：写总结
            summary = self._generate_segment_summary(title, sentences, state.question)
            response_parts.append(f"""<tool_call>
<invoke name="write_segment_summary">
<parameter name="summary">{summary}</parameter>
<parameter name="segment_index">{i}</parameter>
</invoke>
</tool_call>

<tool_result>
<result name="write_segment_summary">
<parameter name="summary">{summary}</parameter>
<parameter name="segment_index">{i}</parameter>
</result>
</tool_result>""")
        
        # 2. 更新当前总结
        current_summary = self._generate_current_summary(state)
        response_parts.append(f"""<tool_call>
<invoke name="update_current_summary">
<parameter name="summary">{current_summary}</parameter>
</invoke>
</tool_call>

<tool_result>
<result name="update_current_summary">
<parameter name="summary">{current_summary}</parameter>
</result>
</tool_result>""")
        
        # 3. 生成最终答案
        final_answer = self._generate_final_answer(state)
        response_parts.append(f"""<tool_call>
<invoke name="generate_final_answer">
<parameter name="answer">{final_answer}</parameter>
</invoke>
</tool_call>

<tool_result>
<result name="generate_final_answer">
<parameter name="answer">{final_answer}</parameter>
</result>
</tool_result>""")
        
        return "\n\n".join(response_parts)
    
    def _generate_segment_summary(self, title: str, sentences: List[str], question: str) -> str:
        """生成段落总结"""
        content = "。".join(sentences)
        
        # 简单的基于关键词的总结
        question_keywords = set(question.lower().split())
        content_words = content.lower().split()
        
        # 找出与问题相关的句子
        relevant_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            if sentence_words & question_keywords:
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            summary = f"关于{title}，主要内容：{'。'.join(relevant_sentences[:2])}。"
        else:
            summary = f"关于{title}，主要内容：{content[:100]}..."
        
        return summary
    
    def _generate_current_summary(self, state: SegmentedReadingState) -> str:
        """生成当前总结"""
        if not state.read_segments:
            return "尚未阅读任何段落。"
        
        summary_parts = []
        for segment_idx in state.read_segments:
            if segment_idx < len(state.context):
                title, sentences = state.context[segment_idx]
                summary_parts.append(f"【{title}】：{'。'.join(sentences[:2])}")
        
        return "；".join(summary_parts) + "。"
    
    def _generate_final_answer(self, state: SegmentedReadingState) -> str:
        """生成最终答案"""
        # 基于支持事实生成答案
        if state.supporting_facts:
            # 找出支持事实对应的内容
            supporting_content = []
            for title, sent_id in state.supporting_facts:
                for segment_idx, (seg_title, sentences) in enumerate(state.context):
                    if seg_title == title and sent_id < len(sentences):
                        supporting_content.append(sentences[sent_id])
            
            if supporting_content:
                return f"根据文档内容，答案是：{'。'.join(supporting_content[:2])}"
        
        # 如果没有找到支持事实，返回一个通用答案
        return f"根据阅读的文档内容，答案是：{state.answer}"
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批处理函数"""
        # 获取最大长度
        max_prompt_len = max(len(item["prompt_input_ids"]) for item in batch)
        max_response_len = max(len(item["response_input_ids"]) for item in batch)
        
        # 填充
        batch_prompt_input_ids = []
        batch_prompt_attention_mask = []
        batch_response_input_ids = []
        batch_response_attention_mask = []
        
        for item in batch:
            # 填充prompt
            prompt_padding_len = max_prompt_len - len(item["prompt_input_ids"])
            batch_prompt_input_ids.append(
                torch.cat([item["prompt_input_ids"], 
                          torch.zeros(prompt_padding_len, dtype=torch.long)])
            )
            batch_prompt_attention_mask.append(
                torch.cat([item["prompt_attention_mask"], 
                          torch.zeros(prompt_padding_len, dtype=torch.long)])
            )
            
            # 填充response
            response_padding_len = max_response_len - len(item["response_input_ids"])
            batch_response_input_ids.append(
                torch.cat([item["response_input_ids"], 
                          torch.zeros(response_padding_len, dtype=torch.long)])
            )
            batch_response_attention_mask.append(
                torch.cat([item["response_attention_mask"], 
                          torch.zeros(response_padding_len, dtype=torch.long)])
            )
        
        return {
            "prompt_input_ids": torch.stack(batch_prompt_input_ids),
            "prompt_attention_mask": torch.stack(batch_prompt_attention_mask),
            "response_input_ids": torch.stack(batch_response_input_ids),
            "response_attention_mask": torch.stack(batch_response_attention_mask),
            "prompt": [item["prompt"] for item in batch],
            "response": [item["response"] for item in batch],
            "sample_ids": [item["sample_id"] for item in batch],
            "questions": [item["question"] for item in batch],
            "answers": [item["answer"] for item in batch],
            "supporting_facts": [item["supporting_facts"] for item in batch]
        }
