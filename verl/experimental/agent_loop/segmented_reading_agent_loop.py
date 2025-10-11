"""
分段阅读Agent Loop
使用文件操作工具进行分段阅读和总结更新
"""

import logging
import asyncio
import os
import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.tools.base_tool import BaseTool
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.tools.segmented_reading_tools import (
    ReadSegmentFileTool,
    WriteSummaryFileTool,
    ReadSummaryFileTool,
    GetDocumentInfoTool
)

logger = logging.getLogger(__name__)


class SegmentedReadingAgentLoop(AgentLoopBase):
    """分段阅读Agent Loop - 使用文件操作工具"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        print("=== DEBUG: SegmentedReadingAgentLoop 构造函数被调用 ===")
        print(f"=== DEBUG: 配置内容: {config} ===")
        
        self.max_segments_to_read = config.get('max_segments_to_read', 10)
        self.segment_length = config.get('segment_length', 2048)
        self.tools_config_file = config.get('tools_config_file')
        
        print(f"=== DEBUG: max_segments_to_read={self.max_segments_to_read}, segment_length={self.segment_length} ===")
        
        self.tools = self._initialize_tools()
        
    def _initialize_tools(self) -> Dict[str, BaseTool]:
        """初始化文件操作工具"""
        tools = {}
        
        # 创建工具实例
        tools["read_segment_file"] = ReadSegmentFileTool()
        tools["write_summary_file"] = WriteSummaryFileTool()
        tools["read_summary_file"] = ReadSummaryFileTool()
        tools["get_document_info"] = GetDocumentInfoTool()
        
        return tools
    
    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """运行分段阅读pipeline"""
        
        # 获取输入数据
        document_file = kwargs.get("document_file", "")
        if not document_file or not os.path.exists(document_file):
            return await self._create_error_output("错误：文档文件不存在")
        
        # 记录Agent Loop开始 - 使用多种输出方式确保可见
        import logging
        import sys
        
        # 1. 使用logging
        logging.getLogger(__name__).info("=== SegmentedReadingAgentLoop Started ===")
        logging.getLogger(__name__).info(f"Document: {document_file}")
        logging.getLogger(__name__).info("=" * 50)
        
        # 2. 强制输出到stderr
        print("=" * 80, file=sys.stderr)
        print("🚀 SEGMENTED READING AGENT LOOP STARTED!", file=sys.stderr)
        print(f"📄 Document: {document_file}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        
        # 读取文档信息
        doc_info_response, _, _ = await self.tools["get_document_info"].execute(
            "doc_info_instance",
            {"file_path": document_file}
        )
        
        if "错误" in doc_info_response.text:
            return await self._create_error_output(doc_info_response.text)
        
        # 解析文档信息
        question = self._extract_question_from_info(doc_info_response.text)
        num_segments = self._extract_num_segments_from_info(doc_info_response.text)
        
        # 初始化总结文件路径
        summary_file = document_file.replace(".json", "_summary.txt")
        
        # 记录解析结果 - 使用多种输出方式确保可见
        logging.getLogger(__name__).info(f"Question: {question}")
        logging.getLogger(__name__).info(f"Total segments: {num_segments}")
        logging.getLogger(__name__).info(f"Summary file: {summary_file}")
        logging.getLogger(__name__).info("=" * 50)
        
        # # 强制输出到stderr
        # print(f"❓ Question: {question}", file=sys.stderr)
        # print(f"📊 Total segments: {num_segments}", file=sys.stderr)
        # print(f"📝 Summary file: {summary_file}", file=sys.stderr)
        # print("=" * 80, file=sys.stderr)
        
        # 初始化状态
        current_segment_index = 0
        segments_read = 0
        turns = 0
        
        # 分段阅读循环
        while segments_read < self.max_segments_to_read and turns < self.max_turns:
            turns += 1
            
            # 构建当前prompt
            prompt = self._build_prompt(question, current_segment_index, num_segments, summary_file)
            # 使用多种输出方式确保可见
            import logging
            import sys
            
            # 1. 使用logging
            logging.getLogger(__name__).info(f"=== Agent Loop Prompt (Turn {turns}) ===")
            logging.getLogger(__name__).info(prompt)
            logging.getLogger(__name__).info("=" * 50)
            
            # 2. 强制输出到stderr
            print(f"🔄 TURN {turns} PROMPT:", file=sys.stderr)
            print(prompt, file=sys.stderr)
            print("-" * 80, file=sys.stderr)
            
            # 调用模型生成响应
            model_response = await self._generate_response(prompt, sampling_params)
            
            # 解析模型响应，提取工具调用
            tool_calls = self._extract_tool_calls(model_response)
            
            # 执行工具调用
            for tool_call in tool_calls:
                result = await self._execute_tool_call(tool_call, document_file, summary_file)
                if result:
                    segments_read += 1
                    current_segment_index += 1
            
            # 检查是否应该停止
            if self._should_stop_reading(segments_read, turns, num_segments):
                break
        
        # 生成最终答案
        final_answer = await self._generate_final_answer(question, summary_file)
        
        # 创建最终的prompt和response
        final_prompt = self._build_prompt(question, current_segment_index, num_segments, summary_file)
        final_prompt += "\n\n请基于以上信息生成最终答案："
        
        # 记录最终prompt - 使用多种输出方式确保可见
        import logging
        import sys
        
        # 1. 使用logging
        logging.getLogger(__name__).info("=== Final Agent Loop Prompt ===")
        logging.getLogger(__name__).info(final_prompt)
        logging.getLogger(__name__).info("=" * 50)
        
        # 2. 强制输出到stderr
        print("🎯 FINAL PROMPT:", file=sys.stderr)
        print(final_prompt, file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        
        # 调用模型生成最终答案
        final_response = await self._generate_response(final_prompt, sampling_params)
        
        # 获取真实的token ids
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(final_prompt)
        )
        
        response_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(final_response)
        )
        
        response_mask = [1] * len(response_ids)
        
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            num_turns=turns,
            metrics=AgentLoopMetrics(
                generate_sequences=float(segments_read),
                tool_calls=float(turns)
            )
        )
    
    async def _create_error_output(self, error_message: str) -> AgentLoopOutput:
        """创建错误输出"""
        # 为错误消息创建token ids
        error_prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(error_message)
        )
        
        return AgentLoopOutput(
            prompt_ids=error_prompt_ids,
            response_ids=[],  # 空响应
            response_mask=[],
            num_turns=0,
            metrics=AgentLoopMetrics(generate_sequences=0.0, tool_calls=0.0)
        )
    
    def _build_prompt(self, question: str, current_segment_index: int, num_segments: int, summary_file: str) -> str:
        """构建当前轮次的prompt"""
        return f"""请阅读文档并回答问题。

问题：{question}

当前进度：已读取 {current_segment_index} 段，共 {num_segments} 段

请使用以下工具继续阅读文档：

1. 读取下一段文档：
   - 工具：read_segment_file
   - 参数：{{"segment_index": {current_segment_index}, "file_path": "文档文件路径"}}

2. 更新总结：
   - 工具：write_summary_file
   - 参数：{{"summary": "总结内容", "file_path": "{summary_file}"}}

3. 读取当前总结：
   - 工具：read_summary_file
   - 参数：{{"file_path": "{summary_file}"}}

请根据当前进度，选择合适的工具继续阅读。如果已经读取了足够的信息来回答问题，请直接给出答案。

你的下一步行动："""
    
    async def _generate_response(self, prompt: str, sampling_params: dict) -> str:
        """调用真实模型生成响应"""
        try:
            # 生成唯一的请求ID
            request_id = uuid4().hex
            
            # 将prompt转换为token ids
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(prompt)
            )
            
            # 调用模型生成
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params
            )
            
            # 将token ids转换回文本
            response_text = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(output.token_ids)
            )
            
            logger.info(f"模型生成成功，响应长度: {len(response_text)}")
            return response_text
            
        except Exception as e:
            logger.error(f"模型生成失败: {str(e)}")
            # 返回默认响应
            return "让我读取下一段文档。"
    
    def _extract_tool_calls(self, response: str) -> List[Dict]:
        """从模型响应中提取工具调用"""
        tool_calls = []
        
        # 简单的工具调用解析（实际应该使用更复杂的解析逻辑）
        if "read_segment_file" in response:
            # 提取段落索引
            import re
            match = re.search(r'segment_index["\']?\s*:\s*(\d+)', response)
            if match:
                tool_calls.append({
                    "tool": "read_segment_file",
                    "parameters": {
                        "segment_index": int(match.group(1)),
                        "file_path": "文档文件路径"  # 这里需要替换为实际路径
                    }
                })
        
        if "write_summary_file" in response:
            # 提取总结内容
            import re
            match = re.search(r'summary["\']?\s*:\s*["\']([^"\']+)["\']', response)
            if match:
                tool_calls.append({
                    "tool": "write_summary_file",
                    "parameters": {
                        "summary": match.group(1),
                        "file_path": "总结文件路径"  # 这里需要替换为实际路径
                    }
                })
        
        return tool_calls
    
    async def _execute_tool_call(self, tool_call: Dict, document_file: str, summary_file: str) -> bool:
        """执行工具调用"""
        tool_name = tool_call["tool"]
        parameters = tool_call["parameters"]
        
        # 替换文件路径
        if "file_path" in parameters:
            if "文档文件" in str(parameters["file_path"]):
                parameters["file_path"] = document_file
            elif "总结文件" in str(parameters["file_path"]):
                parameters["file_path"] = summary_file
        
        try:
            # 创建工具实例
            instance_id = f"{tool_name}_instance_{uuid4().hex[:8]}"
            await self.tools[tool_name].create(instance_id)
            
            # 执行工具
            response, reward, metrics = await self.tools[tool_name].execute(instance_id, parameters)
            
            # 释放工具实例
            await self.tools[tool_name].release(instance_id)
            
            logger.info(f"工具 {tool_name} 执行成功: {response.text[:100]}...")
            return True
            
        except Exception as e:
            logger.error(f"工具 {tool_name} 执行失败: {str(e)}")
            return False
    
    def _should_stop_reading(self, segments_read: int, turns: int, num_segments: int) -> bool:
        """判断是否应该停止阅读"""
        if segments_read >= self.max_segments_to_read:
            return True
        if turns >= self.max_turns:
            return True
        if segments_read >= num_segments:
            return True
        return False
    
    async def _generate_final_answer(self, question: str, summary_file: str) -> str:
        """生成最终答案"""
        # 读取当前总结
        try:
            instance_id = f"read_summary_final_{uuid4().hex[:8]}"
            await self.tools["read_summary_file"].create(instance_id)
            
            response, _, _ = await self.tools["read_summary_file"].execute(
                instance_id,
                {"file_path": summary_file}
            )
            
            await self.tools["read_summary_file"].release(instance_id)
            
            summary = response.text.replace("当前总结：\n", "")
            
            return f"""基于我阅读的文档内容，我认为{question}的答案是：

{summary}

这是基于我阅读的文档段落得出的结论。"""
            
        except Exception as e:
            return f"生成答案时发生错误: {str(e)}"
    
    def _extract_question_from_info(self, info_text: str) -> str:
        """从文档信息中提取问题"""
        import re
        match = re.search(r'问题：(.+)', info_text)
        return match.group(1) if match else "未知问题"
    
    def _extract_num_segments_from_info(self, info_text: str) -> str:
        """从文档信息中提取段落数量"""
        import re
        match = re.search(r'段落数量：(\d+)', info_text)
        return int(match.group(1)) if match else 0
