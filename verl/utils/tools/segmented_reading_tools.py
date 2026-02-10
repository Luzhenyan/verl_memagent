"""
分段阅读工具实现
真正的文件操作工具，用于读取文档段落和写入总结
"""

import os
import json
from typing import Dict, Any, Optional
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class ReadSegmentFileTool(BaseTool):
    """读取文档段落文件"""
    
    def __init__(self, config: dict = None, tool_schema: OpenAIFunctionToolSchema = None):
        if config is None:
            config = {}
        if tool_schema is None:
            tool_schema = self._create_default_schema()
        super().__init__(config, tool_schema)
    
    def _create_default_schema(self) -> OpenAIFunctionToolSchema:
        """创建默认的工具schema"""
        from verl.tools.schemas import OpenAIFunctionSchema, OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema
        
        properties = {
            "segment_index": OpenAIFunctionPropertySchema(
                type="integer",
                description="Index of the segment to read (0-based)"
            ),
            "file_path": OpenAIFunctionPropertySchema(
                type="string",
                description="Path to the document file"
            )
        }
        
        parameters = OpenAIFunctionParametersSchema(
            type="object",
            properties=properties,
            required=["segment_index", "file_path"]
        )
        
        function = OpenAIFunctionSchema(
            name="read_segment_file",
            description="Read a specific segment from a document file",
            parameters=parameters
        )
        
        return OpenAIFunctionToolSchema(type="function", function=function)
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """执行读取段落操作"""
        try:
            segment_index = parameters.get("segment_index")
            file_path = parameters.get("file_path")
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return ToolResponse(text=f"错误：文件 {file_path} 不存在"), 0.0, {}
            
            # 读取文档文件
            with open(file_path, 'r', encoding='utf-8') as f:
                document_data = json.load(f)
            
            segments = document_data.get('segments', [])
            if not segments:
                return ToolResponse(text="错误：文档文件中没有段落数据"), 0.0, {}
            
            # 检查段落索引是否有效
            if segment_index < 0 or segment_index >= len(segments):
                return ToolResponse(text=f"错误：段落索引 {segment_index} 超出范围 (0-{len(segments)-1})"), 0.0, {}
            
            # 返回段落内容
            segment_content = segments[segment_index]['content']
            response_text = f"段落 {segment_index + 1} 内容：\n{segment_content}"
            
            return ToolResponse(text=response_text), 1.0, {"segment_index": segment_index}
                
        except FileNotFoundError:
            return ToolResponse(text=f"错误：文件 {file_path} 不存在"), 0.0, {}
        except json.JSONDecodeError:
            return ToolResponse(text=f"错误：文件 {file_path} 不是有效的JSON格式"), 0.0, {}
        except Exception as e:
            return ToolResponse(text=f"错误：读取文件时发生异常 - {str(e)}"), 0.0, {}


class WriteSummaryFileTool(BaseTool):
    """写入总结文件"""
    
    def __init__(self, config: dict = None, tool_schema: OpenAIFunctionToolSchema = None):
        if config is None:
            config = {}
        if tool_schema is None:
            tool_schema = self._create_default_schema()
        super().__init__(config, tool_schema)
    
    def _create_default_schema(self) -> OpenAIFunctionToolSchema:
        """创建默认的工具schema"""
        from verl.tools.schemas import OpenAIFunctionSchema, OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema
        
        properties = {
            "summary": OpenAIFunctionPropertySchema(
                type="string",
                description="Summary content to write"
            ),
            "file_path": OpenAIFunctionPropertySchema(
                type="string",
                description="Path to the summary file"
            )
        }
        
        parameters = OpenAIFunctionParametersSchema(
            type="object",
            properties=properties,
            required=["summary", "file_path"]
        )
        
        function = OpenAIFunctionSchema(
            name="write_summary_file",
            description="Write or update the summary to a file",
            parameters=parameters
        )
        
        return OpenAIFunctionToolSchema(type="function", function=function)
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """执行写入总结操作"""
        try:
            summary = parameters.get("summary")
            file_path = parameters.get("file_path")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 写入总结文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            response_text = f"成功：总结已写入到 {file_path}"
            return ToolResponse(text=response_text), 1.0, {"file_path": file_path}
            
        except Exception as e:
            return ToolResponse(text=f"错误：写入文件时发生异常 - {str(e)}"), 0.0, {}


class ReadSummaryFileTool(BaseTool):
    """读取总结文件"""
    
    def __init__(self, config: dict = None, tool_schema: OpenAIFunctionToolSchema = None):
        if config is None:
            config = {}
        if tool_schema is None:
            tool_schema = self._create_default_schema()
        super().__init__(config, tool_schema)
    
    def _create_default_schema(self) -> OpenAIFunctionToolSchema:
        """创建默认的工具schema"""
        from verl.tools.schemas import OpenAIFunctionSchema, OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema
        
        properties = {
            "file_path": OpenAIFunctionPropertySchema(
                type="string",
                description="Path to the summary file"
            )
        }
        
        parameters = OpenAIFunctionParametersSchema(
            type="object",
            properties=properties,
            required=["file_path"]
        )
        
        function = OpenAIFunctionSchema(
            name="read_summary_file",
            description="Read the current summary from a file",
            parameters=parameters
        )
        
        return OpenAIFunctionToolSchema(type="function", function=function)
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """执行读取总结操作"""
        try:
            file_path = parameters.get("file_path")
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return ToolResponse(text="无总结内容"), 0.0, {}
            
            # 读取总结文件
            with open(file_path, 'r', encoding='utf-8') as f:
                summary = f.read()
            
            response_text = f"当前总结：\n{summary}"
            return ToolResponse(text=response_text), 1.0, {"file_path": file_path}
            
        except Exception as e:
            return ToolResponse(text=f"错误：读取总结文件时发生异常 - {str(e)}"), 0.0, {}


class GetDocumentInfoTool(BaseTool):
    """获取文档信息"""
    
    def __init__(self, config: dict = None, tool_schema: OpenAIFunctionToolSchema = None):
        if config is None:
            config = {}
        if tool_schema is None:
            tool_schema = self._create_default_schema()
        super().__init__(config, tool_schema)
    
    def _create_default_schema(self) -> OpenAIFunctionToolSchema:
        """创建默认的工具schema"""
        from verl.tools.schemas import OpenAIFunctionSchema, OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema
        
        properties = {
            "file_path": OpenAIFunctionPropertySchema(
                type="string",
                description="Path to the document file"
            )
        }
        
        parameters = OpenAIFunctionParametersSchema(
            type="object",
            properties=properties,
            required=["file_path"]
        )
        
        function = OpenAIFunctionSchema(
            name="get_document_info",
            description="Get basic information about the document",
            parameters=parameters
        )
        
        return OpenAIFunctionToolSchema(type="function", function=function)
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """执行获取文档信息操作"""
        try:
            file_path = parameters.get("file_path")
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return ToolResponse(text=f"错误：文件 {file_path} 不存在"), 0.0, {}
            
            # 读取文档文件
            with open(file_path, 'r', encoding='utf-8') as f:
                document_data = json.load(f)
            
            question = document_data.get('question', '未知问题')
            num_segments = document_data.get('num_segments', 0)
            
            response_text = f"文档信息：\n问题：{question}\n段落数量：{num_segments}"
            return ToolResponse(text=response_text), 1.0, {"question": question, "num_segments": num_segments}
            
        except Exception as e:
            return ToolResponse(text=f"错误：获取文档信息时发生异常 - {str(e)}"), 0.0, {}
