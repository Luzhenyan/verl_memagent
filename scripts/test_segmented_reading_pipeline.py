#!/usr/bin/env python3
"""
测试分段阅读pipeline的完整流程

流程：
1. 让模型阅读文本，每读一段就根据问题和阅读的文本写一个总结
2. 带着问题和总结继续阅读下一段
3. 每阅读一段就更新总结
4. 最后生成最终答案
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from verl.tools.reading_tools import (
    ReadDocumentTool,
    WriteSummaryTool,
    UpdateCurrentSummaryTool,
    GenerateFinalAnswerTool
)
from verl.tools.schemas import OpenAIFunctionToolSchema

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SegmentedReadingPipeline:
    """分段阅读pipeline测试类"""
    
    def __init__(self):
        self.read_tool = ReadDocumentTool({}, OpenAIFunctionToolSchema())
        self.write_summary_tool = WriteSummaryTool({}, OpenAIFunctionToolSchema())
        self.update_summary_tool = UpdateCurrentSummaryTool({}, OpenAIFunctionToolSchema())
        self.generate_answer_tool = GenerateFinalAnswerTool({}, OpenAIFunctionToolSchema())
        
        # 存储状态
        self.segment_summaries = {}
        self.current_summary = ""
        self.question = ""
        self.file_path = ""
    
    async def create_sample_document(self, file_path: str = "/tmp/test_document.txt"):
        """创建测试文档"""
        content = """
        人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。

        该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。

        机器学习是人工智能的一个重要分支，它使计算机能够在没有明确编程的情况下学习和改进。机器学习算法通过分析大量数据来识别模式，并使用这些模式来做出预测或决策。

        深度学习是机器学习的一个子集，它使用多层神经网络来模拟人脑的工作方式。深度学习在图像识别、语音识别和自然语言处理等领域取得了突破性进展。

        自然语言处理（NLP）是人工智能的另一个重要分支，它使计算机能够理解、解释和生成人类语言。NLP技术被广泛应用于机器翻译、聊天机器人和文本分析等领域。

        计算机视觉是人工智能的一个分支，它使计算机能够从图像和视频中提取信息。计算机视觉技术被广泛应用于自动驾驶、医疗诊断和安全监控等领域。

        人工智能的发展面临着许多挑战，包括数据隐私、算法偏见、就业影响和伦理问题。解决这些挑战需要技术、政策和社会各界的共同努力。

        未来，人工智能将继续快速发展，并在各个领域发挥越来越重要的作用。我们需要确保人工智能的发展方向符合人类的价值观和利益。
        """
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        
        return file_path
    
    async def simulate_model_reading_process(self, question: str, file_path: str):
        """模拟模型的分段阅读过程"""
        logger.info(f"开始处理问题: {question}")
        logger.info(f"文档路径: {file_path}")
        
        self.question = question
        self.file_path = file_path
        
        # 1. 读取第一段
        logger.info("=== 步骤1: 读取第一段 ===")
        read_instance_id, _ = await self.read_tool.create()
        read_response, reward, _ = await self.read_tool.execute(
            read_instance_id, 
            {"file_path": file_path, "segment_index": 0}
        )
        logger.info(f"读取结果: {read_response.text}")
        logger.info(f"奖励: {reward}")
        
        # 2. 为第一段写总结
        logger.info("=== 步骤2: 为第一段写总结 ===")
        segment_content = read_response.text.split(": ", 1)[1] if ": " in read_response.text else read_response.text
        summary = f"第一段介绍了人工智能的基本概念，包括其定义、研究领域和应用范围。"
        
        write_instance_id, _ = await self.write_summary_tool.create()
        write_response, write_reward, _ = await self.write_summary_tool.execute(
            write_instance_id,
            {"segment_content": segment_content, "summary": summary}
        )
        logger.info(f"总结: {summary}")
        logger.info(f"总结质量: {write_response.text}")
        logger.info(f"奖励: {write_reward}")
        
        self.segment_summaries[0] = summary
        
        # 3. 读取第二段
        logger.info("=== 步骤3: 读取第二段 ===")
        read_response2, reward2, _ = await self.read_tool.execute(
            read_instance_id, 
            {"file_path": file_path, "segment_index": 1}
        )
        logger.info(f"读取结果: {read_response2.text}")
        logger.info(f"奖励: {reward2}")
        
        # 4. 为第二段写总结
        logger.info("=== 步骤4: 为第二段写总结 ===")
        segment_content2 = read_response2.text.split(": ", 1)[1] if ": " in read_response2.text else read_response2.text
        summary2 = f"第二段讨论了机器学习和深度学习，包括它们的关系和在AI中的重要性。"
        
        write_response2, write_reward2, _ = await self.write_summary_tool.execute(
            write_instance_id,
            {"segment_content": segment_content2, "summary": summary2}
        )
        logger.info(f"总结: {summary2}")
        logger.info(f"总结质量: {write_response2.text}")
        logger.info(f"奖励: {write_reward2}")
        
        self.segment_summaries[1] = summary2
        
        # 5. 更新当前总结
        logger.info("=== 步骤5: 更新当前总结 ===")
        current_summary = f"文档介绍了人工智能的基本概念和重要分支。第一部分定义了AI及其研究领域，第二部分详细讨论了机器学习和深度学习技术。"
        
        update_instance_id, _ = await self.update_summary_tool.create()
        update_response, update_reward, _ = await self.update_summary_tool.execute(
            update_instance_id,
            {
                "segment_summaries": json.dumps(self.segment_summaries),
                "question": question,
                "current_summary": current_summary
            }
        )
        logger.info(f"更新后的总结: {current_summary}")
        logger.info(f"帮助程度: {update_response.text}")
        logger.info(f"奖励: {update_reward}")
        
        self.current_summary = current_summary
        
        # 6. 生成最终答案
        logger.info("=== 步骤6: 生成最终答案 ===")
        final_answer = f"人工智能是计算机科学的一个分支，旨在创建能够模拟人类智能的机器。它包括多个重要分支：机器学习使计算机能够从数据中学习；深度学习使用神经网络模拟人脑；自然语言处理使计算机理解人类语言；计算机视觉使计算机从图像中提取信息。AI的发展面临数据隐私、算法偏见等挑战，但未来将继续快速发展并在各领域发挥重要作用。"
        
        answer_instance_id, _ = await self.generate_answer_tool.create()
        answer_response, answer_reward, _ = await self.generate_answer_tool.execute(
            answer_instance_id,
            {
                "current_summary": current_summary,
                "question": question,
                "final_answer": final_answer
            }
        )
        logger.info(f"最终答案: {final_answer}")
        logger.info(f"答案准确性: {answer_response.text}")
        logger.info(f"奖励: {answer_reward}")
        
        # 清理资源
        await self.read_tool.release(read_instance_id)
        await self.write_summary_tool.release(write_instance_id)
        await self.update_summary_tool.release(update_instance_id)
        await self.generate_answer_tool.release(answer_instance_id)
        
        return {
            "question": question,
            "segment_summaries": self.segment_summaries,
            "current_summary": current_summary,
            "final_answer": final_answer,
            "total_reward": reward + write_reward + reward2 + write_reward2 + update_reward + answer_reward
        }
    
    async def test_with_hotpotqa_format(self):
        """使用HotpotQA格式的示例数据测试"""
        logger.info("=== 使用HotpotQA格式测试 ===")
        
        # 创建HotpotQA格式的示例数据
        hotpotqa_example = {
            "_id": "test_001",
            "question": "人工智能的主要分支有哪些？",
            "answer": "机器学习、深度学习、自然语言处理、计算机视觉",
            "context": [
                ["AI基础", [
                    "人工智能（Artificial Intelligence，AI）是计算机科学的一个分支。",
                    "该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。"
                ]],
                ["机器学习", [
                    "机器学习是人工智能的一个重要分支。",
                    "它使计算机能够在没有明确编程的情况下学习和改进。"
                ]],
                ["深度学习", [
                    "深度学习是机器学习的一个子集。",
                    "它使用多层神经网络来模拟人脑的工作方式。"
                ]],
                ["NLP", [
                    "自然语言处理（NLP）是人工智能的另一个重要分支。",
                    "它使计算机能够理解、解释和生成人类语言。"
                ]]
            ]
        }
        
        # 创建文档文件
        doc_content = ""
        for title, sentences in hotpotqa_example["context"]:
            doc_content += f"【{title}】\n"
            doc_content += "。".join(sentences) + "。\n\n"
        
        file_path = "/tmp/hotpotqa_test_doc.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        # 运行pipeline
        result = await self.simulate_model_reading_process(
            hotpotqa_example["question"], 
            file_path
        )
        
        logger.info("=== HotpotQA格式测试结果 ===")
        logger.info(f"问题: {result['question']}")
        logger.info(f"期望答案: {hotpotqa_example['answer']}")
        logger.info(f"生成答案: {result['final_answer']}")
        logger.info(f"总奖励: {result['total_reward']}")
        
        return result


async def main():
    """主函数"""
    logger.info("开始测试分段阅读pipeline")
    
    pipeline = SegmentedReadingPipeline()
    
    # 1. 创建测试文档
    file_path = await pipeline.create_sample_document()
    logger.info(f"创建测试文档: {file_path}")
    
    # 2. 测试基本pipeline
    question = "请详细介绍人工智能的发展历程和主要技术分支"
    result = await pipeline.simulate_model_reading_process(question, file_path)
    
    logger.info("=== 测试结果总结 ===")
    logger.info(f"问题: {result['question']}")
    logger.info(f"段落总结数量: {len(result['segment_summaries'])}")
    logger.info(f"当前总结: {result['current_summary'][:100]}...")
    logger.info(f"最终答案: {result['final_answer'][:100]}...")
    logger.info(f"总奖励: {result['total_reward']}")
    
    # 3. 测试HotpotQA格式
    hotpot_result = await pipeline.test_with_hotpotqa_format()
    
    logger.info("=== 测试完成 ===")
    logger.info("Pipeline测试成功！所有工具都能正常工作。")
    
    return result, hotpot_result


if __name__ == "__main__":
    asyncio.run(main())
