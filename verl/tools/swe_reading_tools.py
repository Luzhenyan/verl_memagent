# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import json
import asyncio
from typing import Any, Optional, List, Dict
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SWEEnvironment:
    """SWE环境模拟类，提供代码执行和文件操作能力"""
    
    def __init__(self):
        self.workspace = "/tmp/swe_workspace"
        os.makedirs(self.workspace, exist_ok=True)
    
    def read_file(self, file_path: str) -> str:
        """使用SWE读取文件"""
        try:
            # 模拟SWE的文件读取能力
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"SWE read_file error: {e}")
            return ""
    
    def write_file(self, file_path: str, content: str) -> bool:
        """使用SWE写入文件"""
        try:
            # 模拟SWE的文件写入能力
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"SWE write_file error: {e}")
            return False
    
    def run_code(self, code: str, context: dict = None) -> Any:
        """使用SWE执行代码"""
        try:
            # 模拟SWE的代码执行能力
            local_vars = context or {}
            exec(code, globals(), local_vars)
            return local_vars.get('result', None)
        except Exception as e:
            logger.error(f"SWE run_code error: {e}")
            return None
    
    def run_script(self, script_path: str, parameters: dict = None) -> Any:
        """使用SWE运行脚本"""
        try:
            # 模拟SWE的脚本执行能力
            script_content = self.read_file(script_path)
            if script_content:
                context = parameters or {}
                context['result'] = None
                return self.run_code(script_content, context)
            return None
        except Exception as e:
            logger.error(f"SWE run_script error: {e}")
            return None


class SWEReadDocumentTool(BaseTool):
    """基于SWE的文档读取工具"""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.swe = SWEEnvironment()
        self._instance_dict = {}
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "swe_read_document",
                "description": "Use SWE to read and segment a document",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the document file",
                        },
                        "segment_index": {
                            "type": "integer",
                            "description": "Index of the segment to read (0-based)",
                        },
                        "segment_method": {
                            "type": "string",
                            "description": "Method for segmentation (simple/semantic)",
                            "enum": ["simple", "semantic"],
                        },
                    },
                    "required": ["file_path", "segment_index"],
                },
            }
        })
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        
        self._instance_dict[instance_id] = {
            "file_path": "",
            "segments": [],
            "current_segment": "",
            "segment_index": -1,
            "swe_context": {},
        }
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        file_path = parameters.get("file_path", "")
        segment_index = parameters.get("segment_index", 0)
        segment_method = parameters.get("segment_method", "simple")
        
        # 使用SWE读取文档
        if file_path != self._instance_dict[instance_id]["file_path"]:
            content = self.swe.read_file(file_path)
            
            # 使用SWE进行智能分段
            segments = await self._segment_document_with_swe(content, segment_method)
            
            self._instance_dict[instance_id]["file_path"] = file_path
            self._instance_dict[instance_id]["segments"] = segments
        
        # 获取指定段落
        segments = self._instance_dict[instance_id]["segments"]
        if 0 <= segment_index < len(segments):
            current_segment = segments[segment_index]
            self._instance_dict[instance_id]["current_segment"] = current_segment
            self._instance_dict[instance_id]["segment_index"] = segment_index
            
            return ToolResponse(text=f"SWE read segment {segment_index}: {current_segment}"), 0.0, {}
        else:
            return ToolResponse(text=f"Segment index {segment_index} out of range"), -0.1, {}
    
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0
    
    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
    
    async def _segment_document_with_swe(self, content: str, method: str) -> List[str]:
        """使用SWE进行文档分段"""
        if method == "semantic":
            # 使用SWE运行语义分段脚本
            segment_script = """
import re
from typing import List

def semantic_segment(text: str) -> List[str]:
    # 基于语义的分段逻辑
    sentences = re.split(r'[。！？]', text)
    segments = []
    current_segment = ""
    
    for sentence in sentences:
        if len(current_segment) + len(sentence) < 500:
            current_segment += sentence + "。"
        else:
            if current_segment:
                segments.append(current_segment.strip())
            current_segment = sentence + "。"
    
    if current_segment:
        segments.append(current_segment.strip())
    
    result = segments
"""
            return self.swe.run_code(segment_script, {"text": content}) or []
        else:
            # 简单分段
            return self._simple_segment(content)
    
    def _simple_segment(self, content: str) -> List[str]:
        """简单分段"""
        sentences = content.split('。')
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            if len(current_segment) + len(sentence) < 500:
                current_segment += sentence + "。"
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = sentence + "。"
        
        if current_segment:
            segments.append(current_segment.strip())
        
        return segments


class SWEWriteSummaryTool(BaseTool):
    """基于SWE的总结生成工具"""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.swe = SWEEnvironment()
        self._instance_dict = {}
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "swe_write_summary",
                "description": "Use SWE to generate a summary for document content",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Content to summarize",
                        },
                        "summary_style": {
                            "type": "string",
                            "description": "Style of summary (concise/detailed)",
                            "enum": ["concise", "detailed"],
                        },
                    },
                    "required": ["content"],
                },
            }
        })
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        
        self._instance_dict[instance_id] = {
            "content": "",
            "summary": "",
            "quality_score": 0.0,
        }
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        content = parameters.get("content", "")
        summary_style = parameters.get("summary_style", "concise")
        
        # 使用SWE生成总结
        summary = await self._generate_summary_with_swe(content, summary_style)
        
        self._instance_dict[instance_id]["content"] = content
        self._instance_dict[instance_id]["summary"] = summary
        
        # 计算总结质量
        quality_score = self._evaluate_summary_quality(summary, content)
        self._instance_dict[instance_id]["quality_score"] = quality_score
        
        # 只有质量超过阈值才给奖励
        reward = quality_score * 3 if quality_score > 0.5 else 0.0
        
        return ToolResponse(text=f"SWE generated summary with quality score: {quality_score:.2f}"), reward, {}
    
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict[instance_id]["quality_score"]
    
    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
    
    async def _generate_summary_with_swe(self, content: str, style: str) -> str:
        """使用SWE生成总结"""
        # 使用SWE运行总结生成脚本
        summary_script = f"""
import re
from typing import List

def extract_key_points(text: str, style: str) -> str:
    # 提取关键信息
    sentences = re.split(r'[。！？]', text)
    key_sentences = []
    
    for sentence in sentences:
        if len(sentence.strip()) > 10:
            # 简单的关键词提取
            if any(keyword in sentence for keyword in ['主要', '重要', '关键', '核心', '发展', '技术']):
                key_sentences.append(sentence.strip())
    
    if style == "concise":
        # 简洁风格
        result = "。".join(key_sentences[:3]) + "。"
    else:
        # 详细风格
        result = "。".join(key_sentences) + "。"
    
    return result if result else "内容总结"
"""
        result = self.swe.run_code(summary_script, {"text": content, "style": style})
        return result if result else "内容总结"
    
    def _evaluate_summary_quality(self, summary: str, original_content: str) -> float:
        """评估总结质量"""
        if len(summary) < 10:
            return 0.0
        
        # 使用SWE进行质量评估
        quality_script = """
def evaluate_quality(summary: str, original: str) -> float:
    # 简单的质量评估
    if len(summary) < 10:
        return 0.0
    
    # 关键词匹配
    key_words = [word for word in original.split() if len(word) > 3][:10]
    if not key_words:
        return 0.5
    
    matched = sum(1 for word in key_words if word.lower() in summary.lower())
    return min(matched / len(key_words), 1.0)
"""
        result = self.swe.run_code(quality_script, {"summary": summary, "original": original_content})
        return result if result is not None else 0.5


class SWEUpdateSummaryTool(BaseTool):
    """基于SWE的综合总结更新工具"""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.swe = SWEEnvironment()
        self._instance_dict = {}
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "swe_update_summary",
                "description": "Use SWE to update comprehensive summary based on segment summaries",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "segment_summaries": {
                            "type": "string",
                            "description": "JSON string of segment summaries",
                        },
                        "question": {
                            "type": "string",
                            "description": "The question to answer",
                        },
                        "current_summary": {
                            "type": "string",
                            "description": "Current comprehensive summary",
                        },
                    },
                    "required": ["segment_summaries", "question", "current_summary"],
                },
            }
        })
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        
        self._instance_dict[instance_id] = {
            "segment_summaries": {},
            "question": "",
            "current_summary": "",
            "helpfulness_score": 0.0,
        }
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        segment_summaries_str = parameters.get("segment_summaries", "{}")
        question = parameters.get("question", "")
        current_summary = parameters.get("current_summary", "")
        
        try:
            segment_summaries = json.loads(segment_summaries_str)
        except json.JSONDecodeError:
            segment_summaries = {}
        
        # 使用SWE更新总结
        updated_summary = await self._update_summary_with_swe(segment_summaries, question, current_summary)
        
        self._instance_dict[instance_id]["segment_summaries"] = segment_summaries
        self._instance_dict[instance_id]["question"] = question
        self._instance_dict[instance_id]["current_summary"] = updated_summary
        
        # 计算帮助程度
        helpfulness_score = self._evaluate_helpfulness(updated_summary, question)
        self._instance_dict[instance_id]["helpfulness_score"] = helpfulness_score
        
        reward = helpfulness_score * 5
        
        return ToolResponse(text=f"SWE updated summary with helpfulness score: {helpfulness_score:.2f}"), reward, {}
    
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict[instance_id]["helpfulness_score"]
    
    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
    
    async def _update_summary_with_swe(self, segment_summaries: dict, question: str, current_summary: str) -> str:
        """使用SWE更新总结"""
        # 使用SWE运行总结更新脚本
        update_script = """
import json
from typing import Dict, List

def update_comprehensive_summary(segments: Dict, question: str, current: str) -> str:
    # 基于段落总结和问题更新综合总结
    summaries = list(segments.values())
    
    # 提取与问题相关的关键信息
    question_words = set(question.lower().split())
    relevant_info = []
    
    for summary in summaries:
        summary_words = set(summary.lower().split())
        if question_words & summary_words:
            relevant_info.append(summary)
    
    if relevant_info:
        result = "。".join(relevant_info) + "。"
    else:
        result = current if current else "。".join(summaries) + "。"
    
    return result
"""
        result = self.swe.run_code(update_script, {
            "segments": segment_summaries,
            "question": question,
            "current": current_summary
        })
        return result if result else current_summary
    
    def _evaluate_helpfulness(self, summary: str, question: str) -> float:
        """评估总结对回答问题的帮助程度"""
        if len(summary) < 10:
            return 0.0
        
        # 使用SWE进行评估
        helpfulness_script = """
def evaluate_helpfulness(summary: str, question: str) -> float:
    if len(summary) < 10:
        return 0.0
    
    # 检查总结是否包含回答问题所需的信息
    question_words = set(question.lower().split())
    summary_words = set(summary.lower().split())
    
    if not question_words:
        return 0.5
    
    relevance = len(question_words & summary_words) / len(question_words)
    return min(relevance, 1.0)
"""
        result = self.swe.run_code(helpfulness_script, {"summary": summary, "question": question})
        return result if result is not None else 0.5


class SWEGenerateAnswerTool(BaseTool):
    """基于SWE的最终答案生成工具"""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.swe = SWEEnvironment()
        self._instance_dict = {}
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "swe_generate_answer",
                "description": "Use SWE to generate final answer based on summary",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Comprehensive summary",
                        },
                        "question": {
                            "type": "string",
                            "description": "The question to answer",
                        },
                        "final_answer": {
                            "type": "string",
                            "description": "Final answer to the question",
                        },
                    },
                    "required": ["summary", "question", "final_answer"],
                },
            }
        })
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        
        self._instance_dict[instance_id] = {
            "summary": "",
            "question": "",
            "final_answer": "",
            "accuracy_score": 0.0,
        }
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        summary = parameters.get("summary", "")
        question = parameters.get("question", "")
        final_answer = parameters.get("final_answer", "")
        
        # 使用SWE生成答案
        generated_answer = await self._generate_answer_with_swe(summary, question)
        
        self._instance_dict[instance_id]["summary"] = summary
        self._instance_dict[instance_id]["question"] = question
        self._instance_dict[instance_id]["final_answer"] = generated_answer or final_answer
        
        # 计算答案准确性
        accuracy_score = self._evaluate_accuracy(generated_answer or final_answer, question)
        self._instance_dict[instance_id]["accuracy_score"] = accuracy_score
        
        reward = accuracy_score * 10
        
        return ToolResponse(text=f"SWE generated answer with accuracy score: {accuracy_score:.2f}"), reward, {}
    
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict[instance_id]["accuracy_score"]
    
    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
    
    async def _generate_answer_with_swe(self, summary: str, question: str) -> str:
        """使用SWE生成答案"""
        # 使用SWE运行答案生成脚本
        answer_script = """
def generate_answer(summary: str, question: str) -> str:
    # 基于总结和问题生成答案
    if not summary or not question:
        return "无法生成答案"
    
    # 简单的答案生成逻辑
    question_type = "what" if "什么" in question else "how" if "如何" in question else "general"
    
    if question_type == "what":
        # 提取关键信息
        key_info = summary.split("。")[:3]
        result = "。".join(key_info) + "。"
    else:
        result = summary
    
    return result
"""
        result = self.swe.run_code(answer_script, {"summary": summary, "question": question})
        return result if result else "基于总结生成的答案"
    
    def _evaluate_accuracy(self, answer: str, question: str) -> float:
        """评估答案准确性"""
        if len(answer) < 10:
            return 0.0
        
        # 使用SWE进行评估
        accuracy_script = """
def evaluate_accuracy(answer: str, question: str) -> float:
    if len(answer) < 10:
        return 0.0
    
    # 检查答案是否包含问题的关键词
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())
    
    if not question_words:
        return 0.5
    
    relevance = len(question_words & answer_words) / len(question_words)
    length_factor = min(len(answer) / 50, 1.0)
    
    return min((relevance + length_factor) / 2, 1.0)
"""
        result = self.swe.run_code(accuracy_script, {"answer": answer, "question": question})
        return result if result is not None else 0.5
