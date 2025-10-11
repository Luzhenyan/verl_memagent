#!/usr/bin/env python3
"""
基于Transformers的本地7B模型调用器
参考SWE-agent的方式调用我们的分段阅读工具
"""

import asyncio
import json
import logging
import sys
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from verl.tools.reading_tools import (
    ReadDocumentTool,
    WriteSummaryTool,
    UpdateCurrentSummaryTool,
    GenerateFinalAnswerTool
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalModelWithTools:
    """基于Transformers的本地模型工具调用器"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.tools = {}
        self.tool_schemas = []
        
        # 初始化工具
        self._init_tools()
        
        # 加载模型
        self._load_model()
    
    def _init_tools(self):
        """初始化工具"""
        logger.info("初始化分段阅读工具...")
        
        # 创建工具实例
        self.tools = {
            "read_document_segment": ReadDocumentTool({}, None),
            "write_segment_summary": WriteSummaryTool({}, None),
            "update_current_summary": UpdateCurrentSummaryTool({}, None),
            "generate_final_answer": GenerateFinalAnswerTool({}, None)
        }
        
        # 获取工具schema
        self.tool_schemas = [
            self.tools["read_document_segment"].get_openai_tool_schema().model_dump(),
            self.tools["write_segment_summary"].get_openai_tool_schema().model_dump(),
            self.tools["update_current_summary"].get_openai_tool_schema().model_dump(),
            self.tools["generate_final_answer"].get_openai_tool_schema().model_dump()
        ]
        
        logger.info(f"已初始化 {len(self.tools)} 个工具")
        for tool_name in self.tools.keys():
            logger.info(f"  - {tool_name}")
    
    def _load_model(self):
        """加载本地模型"""
        logger.info(f"正在加载模型: {self.model_name}")
        
        try:
            # 配置量化参数（节省内存）
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            logger.info("模型加载成功！")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.info("尝试使用CPU加载...")
            
            # 如果GPU加载失败，尝试CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="cpu",
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            logger.info("模型在CPU上加载成功！")
    
    def _create_tool_prompt(self, messages: List[Dict], tools: List[Dict]) -> str:
        """创建包含工具信息的提示"""
        # 系统提示
        system_prompt = """你是一个智能助手，可以使用以下工具来帮助用户：

可用工具：
"""
        
        # 添加工具描述
        for tool in tools:
            tool_info = tool["function"]
            system_prompt += f"- {tool_info['name']}: {tool_info['description']}\n"
            system_prompt += f"  参数: {json.dumps(tool_info['parameters'], ensure_ascii=False, indent=2)}\n\n"
        
        system_prompt += """
请根据用户的问题，选择合适的工具来完成任务。如果用户的问题需要分段阅读文档，请按照以下步骤进行：
1. 使用 read_document_segment 读取文档段落
2. 使用 write_segment_summary 为段落写总结
3. 使用 update_current_summary 更新当前总结
4. 使用 generate_final_answer 生成最终答案

请以JSON格式返回工具调用，格式如下：
{
  "tool": "工具名称",
  "arguments": {
    "参数名": "参数值"
  }
}
"""
        
        # 构建完整提示
        full_prompt = system_prompt + "\n\n"
        
        # 添加对话历史
        for message in messages:
            if message["role"] == "user":
                full_prompt += f"用户: {message['content']}\n"
            elif message["role"] == "assistant":
                full_prompt += f"助手: {message['content']}\n"
            elif message["role"] == "tool":
                full_prompt += f"工具结果: {message['content']}\n"
        
        full_prompt += "助手: "
        
        return full_prompt
    
    def _parse_tool_call(self, response: str) -> Optional[Dict]:
        """解析模型响应中的工具调用"""
        try:
            # 尝试提取JSON部分
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                tool_call = json.loads(json_str)
                
                # 验证工具调用格式
                if "tool" in tool_call and "arguments" in tool_call:
                    return tool_call
            
        except json.JSONDecodeError:
            logger.warning(f"无法解析工具调用JSON: {response}")
        
        return None
    
    async def call_model_with_tools(self, messages: List[Dict], max_new_tokens: int = 512) -> Dict:
        """调用模型并处理工具调用"""
        logger.info("调用本地模型...")
        
        # 创建提示
        prompt = self._create_tool_prompt(messages, self.tool_schemas)
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码响应
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取新生成的部分
        new_response = response[len(prompt):].strip()
        
        logger.info(f"模型响应: {new_response}")
        
        # 解析工具调用
        tool_call = self._parse_tool_call(new_response)
        
        if tool_call:
            logger.info(f"检测到工具调用: {tool_call['tool']}")
            return {
                "message": new_response,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": tool_call["tool"],
                        "arguments": json.dumps(tool_call["arguments"], ensure_ascii=False)
                    }
                }]
            }
        else:
            logger.info("未检测到工具调用，返回普通响应")
            return {
                "message": new_response,
                "tool_calls": []
            }
    
    async def execute_tool_call(self, tool_call: Dict) -> str:
        """执行工具调用"""
        tool_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])
        
        logger.info(f"执行工具: {tool_name}")
        logger.info(f"参数: {arguments}")
        
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            
            try:
                # 创建工具实例
                instance_id, _ = await tool.create()
                
                # 执行工具
                response, reward, metrics = await tool.execute(instance_id, arguments)
                
                # 释放资源
                await tool.release(instance_id)
                
                logger.info(f"工具执行成功，奖励: {reward}")
                return response.text
                
            except Exception as e:
                logger.error(f"工具执行失败: {e}")
                return f"工具执行失败: {e}"
        else:
            logger.error(f"未知工具: {tool_name}")
            return f"未知工具: {tool_name}"
    
    async def run_conversation(self, initial_message: str, max_turns: int = 10) -> List[Dict]:
        """运行完整的对话流程"""
        messages = [
            {"role": "user", "content": initial_message}
        ]
        
        conversation_history = []
        
        for turn in range(max_turns):
            logger.info(f"=== 第 {turn + 1} 轮对话 ===")
            
            # 调用模型
            response = await self.call_model_with_tools(messages)
            
            # 添加助手响应
            messages.append({
                "role": "assistant",
                "content": response["message"]
            })
            
            conversation_history.append({
                "turn": turn + 1,
                "user_message": messages[-2]["content"],
                "assistant_response": response["message"],
                "tool_calls": response.get("tool_calls", [])
            })
            
            # 如果有工具调用，执行工具
            if response.get("tool_calls"):
                for tool_call in response["tool_calls"]:
                    tool_result = await self.execute_tool_call(tool_call)
                    
                    # 添加工具结果
                    messages.append({
                        "role": "tool",
                        "content": tool_result
                    })
                    
                    conversation_history[-1]["tool_results"] = tool_result
            
            # 检查是否完成
            if "最终答案" in response["message"] or "答案" in response["message"]:
                logger.info("对话完成")
                break
        
        return conversation_history


async def test_local_model_with_hotpotqa():
    """使用HotpotQA数据测试本地模型"""
    logger.info("开始测试本地模型与HotpotQA数据")
    
    # 创建模型实例
    model = LocalModelWithTools()
    
    # 加载HotpotQA数据
    try:
        with open("/home/luzhenyan/datasets/hotpot_dev_distractor_v1.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 选择第一个样本进行测试
        sample = data[0]
        question = sample["question"]
        answer = sample["answer"]
        
        logger.info(f"测试问题: {question}")
        logger.info(f"正确答案: {answer}")
        
        # 创建文档文件
        from verl.scripts.test_hotpotqa_pipeline import create_document_from_context
        doc_content = create_document_from_context(sample["context"])
        file_path = "/tmp/hotpotqa_test_doc.txt"
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc_content)
        
        # 构建初始消息
        initial_message = f"""我需要回答以下问题：{question}

文档已经保存在 {file_path}，请使用分段阅读工具来帮助我回答这个问题。

请按照以下步骤进行：
1. 先读取第一段文档
2. 为第一段写总结
3. 继续读取后续段落并写总结
4. 更新当前总结
5. 生成最终答案"""

        # 运行对话
        conversation = await model.run_conversation(initial_message, max_turns=8)
        
        # 输出结果
        logger.info("=== 对话历史 ===")
        for turn in conversation:
            logger.info(f"轮次 {turn['turn']}:")
            logger.info(f"  用户: {turn['user_message']}")
            logger.info(f"  助手: {turn['assistant_response']}")
            if 'tool_results' in turn:
                logger.info(f"  工具结果: {turn['tool_results']}")
            logger.info("---")
        
        return conversation
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return None


async def main():
    """主函数"""
    logger.info("开始本地模型工具调用测试")
    
    # 测试本地模型
    result = await test_local_model_with_hotpotqa()
    
    if result:
        logger.info("测试完成！")
        logger.info(f"共进行了 {len(result)} 轮对话")
    else:
        logger.error("测试失败")


if __name__ == "__main__":
    asyncio.run(main())
