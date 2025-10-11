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
from typing import Any, Optional, List, Dict
from uuid import uuid4

from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ReadDocumentTool(BaseTool):
    """Tool for reading document segments."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "read_document_segment",
                "description": "Read a specific segment of a document",
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
        }
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        file_path = parameters.get("file_path", "")
        segment_index = parameters.get("segment_index", 0)
        
        # 读取文档并分段
        if file_path != self._instance_dict[instance_id]["file_path"]:
            content = self._read_file(file_path)
            segments = self._segment_document(content)
            self._instance_dict[instance_id]["file_path"] = file_path
            self._instance_dict[instance_id]["segments"] = segments
        
        # 获取指定段落
        segments = self._instance_dict[instance_id]["segments"]
        if 0 <= segment_index < len(segments):
            current_segment = segments[segment_index]
            self._instance_dict[instance_id]["current_segment"] = current_segment
            self._instance_dict[instance_id]["segment_index"] = segment_index
            
            return ToolResponse(text=f"Read segment {segment_index}: {current_segment}"), 0.0, {}
        else:
            return ToolResponse(text=f"Segment index {segment_index} out of range"), -0.1, {}
    
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0
    
    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
    
    def _read_file(self, file_path: str) -> str:
        """Read file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ""
    
    def _segment_document(self, content: str, max_length: int = 500) -> List[str]:
        """Segment document into chunks."""
        sentences = content.split('。')
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            if len(current_segment) + len(sentence) < max_length:
                current_segment += sentence + "。"
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = sentence + "。"
        
        if current_segment:
            segments.append(current_segment.strip())
        
        return segments


class WriteSummaryTool(BaseTool):
    """Tool for writing segment summaries."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "write_segment_summary",
                "description": "Write a summary for a document segment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "segment_content": {
                            "type": "string",
                            "description": "Content of the segment to summarize",
                        },
                        "summary": {
                            "type": "string",
                            "description": "Summary of the segment",
                        },
                    },
                    "required": ["segment_content", "summary"],
                },
            }
        })
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        
        self._instance_dict[instance_id] = {
            "segment_content": "",
            "summary": "",
            "quality_score": 0.0,
        }
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        segment_content = parameters.get("segment_content", "")
        summary = parameters.get("summary", "")
        
        self._instance_dict[instance_id]["segment_content"] = segment_content
        self._instance_dict[instance_id]["summary"] = summary
        
        # 计算总结质量
        quality_score = self._evaluate_summary_quality(summary, segment_content)
        self._instance_dict[instance_id]["quality_score"] = quality_score
        
        # 只有质量超过阈值才给奖励
        reward = quality_score * 3 if quality_score > 0.5 else 0.0
        
        return ToolResponse(text=f"Summary written with quality score: {quality_score:.2f}"), reward, {}
    
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict[instance_id]["quality_score"]
    
    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
    
    def _evaluate_summary_quality(self, summary: str, original_content: str) -> float:
        """Evaluate summary quality."""
        if len(summary) < 10:
            return 0.0
        
        # 简单的关键词匹配评估
        key_words = self._extract_key_words(original_content)
        if not key_words:
            return 0.5  # 默认分数
        
        matched = sum(1 for word in key_words if word.lower() in summary.lower())
        return min(matched / len(key_words), 1.0)
    
    def _extract_key_words(self, content: str) -> List[str]:
        """Extract key words from content."""
        # 简单的关键词提取（可以后续改进）
        words = content.split()
        # 过滤掉短词和常见词
        key_words = [word for word in words if len(word) > 3]
        return key_words[:10]  # 取前10个关键词


class UpdateCurrentSummaryTool(BaseTool):
    """Tool for updating the current summary based on all segment summaries."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "update_current_summary",
                "description": "Update the current summary based on all segment summaries",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "segment_summaries": {
                            "type": "string",
                            "description": "JSON string of segment summaries with indices",
                        },
                        "question": {
                            "type": "string",
                            "description": "The question to answer",
                        },
                        "current_summary": {
                            "type": "string",
                            "description": "Updated current summary",
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
        
        self._instance_dict[instance_id]["segment_summaries"] = segment_summaries
        self._instance_dict[instance_id]["question"] = question
        self._instance_dict[instance_id]["current_summary"] = current_summary
        
        # 计算总结对回答问题的帮助程度
        helpfulness_score = self._evaluate_summary_helpfulness(current_summary, question)
        self._instance_dict[instance_id]["helpfulness_score"] = helpfulness_score
        
        reward = helpfulness_score * 5
        
        return ToolResponse(text=f"Current summary updated with helpfulness score: {helpfulness_score:.2f}"), reward, {}
    
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict[instance_id]["helpfulness_score"]
    
    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
    
    def _evaluate_summary_helpfulness(self, summary: str, question: str) -> float:
        """Evaluate how helpful the summary is for answering the question."""
        if len(summary) < 10:
            return 0.0
        
        # 检查总结是否包含回答问题所需的信息
        question_words = set(question.lower().split())
        summary_words = set(summary.lower().split())
        
        if not question_words:
            return 0.5  # 默认分数
        
        relevance = len(question_words & summary_words) / len(question_words)
        return min(relevance, 1.0)


class GenerateFinalAnswerTool(BaseTool):
    """Tool for generating final answer based on current summary."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
    
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "generate_final_answer",
                "description": "Generate final answer based on current summary",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "current_summary": {
                            "type": "string",
                            "description": "Current summary of all read segments",
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
                    "required": ["current_summary", "question", "final_answer"],
                },
            }
        })
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        
        self._instance_dict[instance_id] = {
            "current_summary": "",
            "question": "",
            "final_answer": "",
            "accuracy_score": 0.0,
        }
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        current_summary = parameters.get("current_summary", "")
        question = parameters.get("question", "")
        final_answer = parameters.get("final_answer", "")
        
        self._instance_dict[instance_id]["current_summary"] = current_summary
        self._instance_dict[instance_id]["question"] = question
        self._instance_dict[instance_id]["final_answer"] = final_answer
        
        # 计算答案准确性
        accuracy_score = self._evaluate_answer_accuracy(final_answer, question)
        self._instance_dict[instance_id]["accuracy_score"] = accuracy_score
        
        reward = accuracy_score * 10
        
        return ToolResponse(text=f"Final answer generated with accuracy score: {accuracy_score:.2f}"), reward, {}
    
    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict[instance_id]["accuracy_score"]
    
    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
    
    def _evaluate_answer_accuracy(self, answer: str, question: str) -> float:
        """Evaluate answer accuracy."""
        if len(answer) < 10:
            return 0.0
        
        # 简单的答案质量评估（可以后续改进）
        # 检查答案是否包含问题的关键词
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        if not question_words:
            return 0.5  # 默认分数
        
        relevance = len(question_words & answer_words) / len(question_words)
        length_factor = min(len(answer) / 50, 1.0)  # 长度因子
        
        return min((relevance + length_factor) / 2, 1.0)
