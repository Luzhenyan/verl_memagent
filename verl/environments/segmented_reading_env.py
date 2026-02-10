"""
SegmentedReadingEnvironment - 分段阅读强化学习环境
基于HotpotQA数据集设计，使用现有工具，重点设计奖励函数
"""

import json
import logging
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


@dataclass
class SegmentedReadingState:
    """分段阅读环境的状态"""
    
    # HotpotQA数据
    question: str = ""
    answer: str = ""
    supporting_facts: List[Tuple[str, int]] = field(default_factory=list)
    context: List[Tuple[str, List[str]]] = field(default_factory=list)
    
    # 当前状态
    current_summary: str = ""
    read_segments: List[int] = field(default_factory=list)
    available_segments: List[int] = field(default_factory=list)
    
    # 环境控制
    current_step: int = 0
    max_steps: int = 20
    
    # 完成状态
    is_done: bool = False
    final_answer: str = ""


class SegmentedReadingEnvironment:
    """分段阅读强化学习环境"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_steps = config.get("max_steps", 20)
        
        # 奖励权重
        self.reward_weights = config.get("reward_weights", {
            "read_segment": 0.1,      # 读取段落基础奖励
            "write_summary": 0.3,     # 写总结奖励
            "update_summary": 0.3,    # 更新总结奖励
            "extract_facts": 0.5,     # 提取关键事实奖励（通过总结内容评估）
            "final_answer": 1.0,      # 最终答案奖励
            "step_penalty": -0.01     # 步骤惩罚
        })
        
        self.current_state: Optional[SegmentedReadingState] = None
        self.hotpotqa_data: List[Dict[str, Any]] = []
        
        logger.info(f"SegmentedReadingEnvironment initialized")
    
    def load_data(self, data_path: str):
        """加载HotpotQA数据"""
        logger.info(f"Loading HotpotQA data from {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.hotpotqa_data = json.load(f)
        
        logger.info(f"Loaded {len(self.hotpotqa_data)} HotpotQA samples")
    
    def reset(self, episode_id: Optional[int] = None) -> SegmentedReadingState:
        """重置环境到初始状态"""
        if not self.hotpotqa_data:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if episode_id is None:
            episode_id = random.randint(0, len(self.hotpotqa_data) - 1)
        
        sample = self.hotpotqa_data[episode_id]
        
        # 支持TriviaQA格式（有segments字段）和HotpotQA格式（有context字段）
        if 'segments' in sample and sample['segments']:
            # TriviaQA格式
            segments = sample['segments']
            context = [(seg['title'], [seg['content']]) for seg in segments]
            supporting_facts = []  # TriviaQA没有supporting_facts
        elif 'context' in sample and sample['context']:
            # HotpotQA格式
            context = sample['context']
            supporting_facts = sample.get('supporting_facts', [])
        else:
            # 默认格式
            context = []
            supporting_facts = []
        
        self.current_state = SegmentedReadingState(
            question=sample['question'],
            answer=sample['answer'],
            supporting_facts=supporting_facts,
            context=context,
            available_segments=list(range(len(context))),
            max_steps=self.max_steps
        )
        
        logger.info(f"Environment reset for episode {episode_id}")
        logger.info(f"Question: {self.current_state.question}")
        logger.info(f"Number of segments: {len(context)}")
        
        return self.current_state
    
    def step(self, action: str, action_args: Dict[str, Any]) -> Tuple[SegmentedReadingState, float, bool, Dict[str, Any]]:
        """执行一步动作"""
        if self.current_state is None:
            raise ValueError("Environment not reset")
        
        if self.current_state.is_done:
            return self.current_state, 0.0, True, {"message": "Episode already done"}
        
        self.current_state.current_step += 1
        reward = 0.0
        info = {}
        
        # 执行动作
        if action == "read_document_segment":
            reward, info = self._execute_read_segment(action_args)
        elif action == "write_segment_summary":
            reward, info = self._execute_write_summary(action_args)
        elif action == "update_current_summary":
            reward, info = self._execute_update_summary(action_args)
        elif action == "generate_final_answer":
            reward, info = self._execute_generate_answer(action_args)
        else:
            info = {"message": f"Unknown action: {action}"}
        
        # 检查完成状态
        done = self._check_done()
        if done:
            self.current_state.is_done = True
        
        # 添加步骤惩罚
        reward += self.reward_weights["step_penalty"]
        
        return self.current_state, reward, done, info
    
    def _execute_read_segment(self, action_args: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """执行读取段落动作"""
        segment_index = action_args.get("segment_index", 0)
        
        if segment_index not in self.current_state.available_segments:
            return -0.1, {"message": f"Segment {segment_index} not available"}
        
        if segment_index < len(self.current_state.context):
            title, sentences = self.current_state.context[segment_index]
            segment_content = f"【{title}】\n" + "。".join(sentences) + "。"
            
            self.current_state.read_segments.append(segment_index)
            self.current_state.available_segments.remove(segment_index)
            
            reward = self.reward_weights["read_segment"]
            info = {
                "message": f"Read segment {segment_index}: {title}",
                "segment_content": segment_content,
                "segment_index": segment_index,
                "title": title
            }
            
            return reward, info
        else:
            return -0.1, {"message": f"Invalid segment index: {segment_index}"}
    
    def _execute_write_summary(self, action_args: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """执行写总结动作 - 重点评估关键事实提取"""
        summary = action_args.get("summary", "")
        segment_index = action_args.get("segment_index", -1)
        
        if segment_index not in self.current_state.read_segments:
            return -0.1, {"message": f"Segment {segment_index} not read yet"}
        
        # 评估总结质量
        quality_score = self._evaluate_summary_quality(summary, segment_index)
        # 评估关键事实提取
        fact_extraction_score = self._evaluate_fact_extraction(summary, segment_index)
        
        # 综合奖励：总结质量 + 事实提取奖励
        total_reward = (quality_score * self.reward_weights["write_summary"] + 
                       fact_extraction_score * self.reward_weights["extract_facts"])
        
        info = {
            "message": f"Wrote summary for segment {segment_index}",
            "summary": summary,
            "quality_score": quality_score,
            "fact_extraction_score": fact_extraction_score,
            "total_reward": total_reward
        }
        
        return total_reward, info
    
    def _execute_update_summary(self, action_args: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """执行更新总结动作 - 评估事实覆盖度"""
        new_summary = action_args.get("summary", "")
        
        # 评估总结相关性
        relevance_score = self._evaluate_summary_relevance(new_summary)
        # 评估事实覆盖度
        fact_coverage_score = self._evaluate_fact_coverage(new_summary)
        
        # 更新总结
        self.current_state.current_summary = new_summary
        
        total_reward = (relevance_score * self.reward_weights["update_summary"] + 
                       fact_coverage_score * self.reward_weights["extract_facts"])
        
        info = {
            "message": "Updated current summary",
            "summary": new_summary,
            "relevance_score": relevance_score,
            "fact_coverage_score": fact_coverage_score,
            "total_reward": total_reward
        }
        
        return total_reward, info
    
    def _execute_generate_answer(self, action_args: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """执行生成答案动作"""
        answer = action_args.get("answer", "")
        
        # 评估答案准确性
        accuracy_score = self._evaluate_answer_accuracy(answer)
        
        # 保存答案
        self.current_state.final_answer = answer
        
        reward = accuracy_score * self.reward_weights["final_answer"]
        info = {
            "message": "Generated final answer",
            "answer": answer,
            "accuracy_score": accuracy_score
        }
        
        return reward, info
    
    def _evaluate_summary_quality(self, summary: str, segment_index: int) -> float:
        """评估段落总结质量"""
        if not summary:
            return 0.0
        
        # 获取段落内容
        title, sentences = self.current_state.context[segment_index]
        segment_content = " ".join(sentences)
        
        # 长度合理性
        length_score = min(len(summary) / 100.0, 1.0)
        
        # 关键词匹配
        segment_words = set(re.findall(r'\w+', segment_content.lower()))
        summary_words = set(re.findall(r'\w+', summary.lower()))
        overlap = len(segment_words.intersection(summary_words))
        keyword_score = min(overlap / max(len(segment_words), 1), 1.0)
        
        # 综合评分
        quality_score = (length_score + keyword_score) / 2.0
        
        return quality_score
    
    def _evaluate_fact_extraction(self, summary: str, segment_index: int) -> float:
        """评估关键事实提取质量 - 核心奖励函数"""
        if not summary:
            return 0.0
        
        # 获取该段落的支持事实
        title, _ = self.current_state.context[segment_index]
        segment_supporting_facts = [
            fact for fact in self.current_state.supporting_facts 
            if fact[0] == title
        ]
        
        if not segment_supporting_facts:
            return 0.0
        
        # 评估总结是否包含了支持事实中的关键信息
        fact_score = 0.0
        for supporting_fact in segment_supporting_facts:
            fact_title = supporting_fact[0]
            # 检查总结中是否提到了支持事实的标题
            if fact_title.lower() in summary.lower():
                fact_score += 0.5
            
            # 检查总结中是否包含了支持事实的关键词
            fact_words = set(re.findall(r'\w+', fact_title.lower()))
            summary_words = set(re.findall(r'\w+', summary.lower()))
            overlap = len(fact_words.intersection(summary_words))
            if overlap > 0:
                fact_score += 0.3
        
        return min(fact_score, 1.0)
    
    def _evaluate_summary_relevance(self, summary: str) -> float:
        """评估总结相关性"""
        if not summary:
            return 0.0
        
        # 检查总结是否包含问题中的关键词
        question_words = set(re.findall(r'\w+', self.current_state.question.lower()))
        summary_words = set(re.findall(r'\w+', summary.lower()))
        
        overlap = len(question_words.intersection(summary_words))
        relevance_score = min(overlap / max(len(question_words), 1), 1.0)
        
        return relevance_score
    
    def _evaluate_fact_coverage(self, summary: str) -> float:
        """评估总结对关键事实的覆盖程度"""
        if not summary:
            return 0.0
        
        # 检查总结是否包含了支持事实中的关键信息
        covered_facts = 0
        total_facts = len(self.current_state.supporting_facts)
        
        for supporting_fact in self.current_state.supporting_facts:
            fact_title = supporting_fact[0]
            if fact_title.lower() in summary.lower():
                covered_facts += 1
        
        if total_facts > 0:
            coverage_score = covered_facts / total_facts
            return coverage_score
        
        return 0.0
    
    def _evaluate_answer_accuracy(self, answer: str) -> float:
        """评估答案准确性"""
        if not answer:
            return 0.0
        
        correct_answer = self.current_state.answer.lower().strip()
        given_answer = answer.lower().strip()
        
        # 完全匹配
        if given_answer == correct_answer:
            return 1.0
        
        # 部分匹配
        if correct_answer in given_answer or given_answer in correct_answer:
            return 0.8
        
        # 关键词匹配
        correct_words = set(re.findall(r'\w+', correct_answer))
        given_words = set(re.findall(r'\w+', given_answer))
        overlap = len(correct_words.intersection(given_words))
        
        if len(correct_words) > 0:
            accuracy_score = overlap / len(correct_words)
            return min(accuracy_score, 0.6)
        
        return 0.0
    
    def _check_done(self) -> bool:
        """检查是否完成"""
        if self.current_state.current_step >= self.max_steps:
            return True
        
        if self.current_state.final_answer:
            return True
        
        return False
    
    def get_state_info(self) -> Dict[str, Any]:
        """获取当前状态信息"""
        if self.current_state is None:
            return {}
        
        return {
            "current_step": self.current_state.current_step,
            "max_steps": self.max_steps,
            "read_segments": len(self.current_state.read_segments),
            "total_segments": len(self.current_state.context),
            "has_final_answer": bool(self.current_state.final_answer),
            "supporting_facts": self.current_state.supporting_facts
        }
    
    def render(self) -> str:
        """渲染当前状态"""
        if self.current_state is None:
            return "Environment not initialized"
        
        state_info = self.get_state_info()
        
        render_str = f"""
=== SegmentedReadingEnvironment State ===
Question: {self.current_state.question}
Answer: {self.current_state.answer}
Current Step: {state_info['current_step']}/{state_info['max_steps']}
Read Segments: {state_info['read_segments']}/{state_info['total_segments']}
Supporting Facts: {self.current_state.supporting_facts}
Current Summary: {self.current_state.current_summary[:100]}...
Final Answer: {self.current_state.final_answer[:100] if self.current_state.final_answer else 'Not generated'}
========================================
"""
        return render_str
