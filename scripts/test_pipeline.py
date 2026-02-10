#!/usr/bin/env python3
"""
测试分段阅读pipeline的完整流程
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from verl.tools.reading_tools import (
    ReadDocumentTool,
    WriteSummaryTool,
    UpdateCurrentSummaryTool,
    GenerateFinalAnswerTool
)
from verl.tools.schemas import OpenAIFunctionToolSchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_pipeline():
    """测试分段阅读pipeline"""
    
    # 初始化工具
    read_tool = ReadDocumentTool({}, None)
    write_tool = WriteSummaryTool({}, None)
    update_tool = UpdateCurrentSummaryTool({}, None)
    answer_tool = GenerateFinalAnswerTool({}, None)
    
    # 创建测试文档
    test_content = """
    人工智能（AI）是计算机科学的一个分支。该领域包括机器人、语言识别、图像识别等。
    机器学习是AI的重要分支，使计算机能够从数据中学习。深度学习使用神经网络模拟人脑。
    自然语言处理使计算机理解人类语言。计算机视觉使计算机从图像中提取信息。
    """
    
    file_path = "/tmp/test_doc.txt"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(test_content.strip())
    
    question = "人工智能的主要分支有哪些？"
    segment_summaries = {}
    
    logger.info("=== 开始测试pipeline ===")
    
    # 1. 读取第一段
    read_id, _ = await read_tool.create()
    read_resp, reward1, _ = await read_tool.execute(read_id, {"file_path": file_path, "segment_index": 0})
    logger.info(f"读取第一段: {read_resp.text}")
    
    # 2. 写第一段总结
    write_id, _ = await write_tool.create()
    summary1 = "第一段介绍了人工智能的基本概念和研究领域。"
    write_resp, reward2, _ = await write_tool.execute(write_id, {"segment_content": "AI概念", "summary": summary1})
    segment_summaries[0] = summary1
    logger.info(f"第一段总结: {summary1}")
    
    # 3. 读取第二段
    read_resp2, reward3, _ = await read_tool.execute(read_id, {"file_path": file_path, "segment_index": 1})
    logger.info(f"读取第二段: {read_resp2.text}")
    
    # 4. 写第二段总结
    summary2 = "第二段讨论了机器学习和深度学习技术。"
    write_resp2, reward4, _ = await write_tool.execute(write_id, {"segment_content": "ML/DL", "summary": summary2})
    segment_summaries[1] = summary2
    logger.info(f"第二段总结: {summary2}")
    
    # 5. 更新当前总结
    update_id, _ = await update_tool.create()
    current_summary = "文档介绍了AI的基本概念，包括机器学习和深度学习。"
    update_resp, reward5, _ = await update_tool.execute(update_id, {
        "segment_summaries": json.dumps(segment_summaries),
        "question": question,
        "current_summary": current_summary
    })
    logger.info(f"当前总结: {current_summary}")
    
    # 6. 生成最终答案
    answer_id, _ = await answer_tool.create()
    final_answer = "人工智能的主要分支包括机器学习、深度学习、自然语言处理和计算机视觉。"
    answer_resp, reward6, _ = await answer_tool.execute(answer_id, {
        "current_summary": current_summary,
        "question": question,
        "final_answer": final_answer
    })
    logger.info(f"最终答案: {final_answer}")
    
    # 清理资源
    await read_tool.release(read_id)
    await write_tool.release(write_id)
    await update_tool.release(update_id)
    await answer_tool.release(answer_id)
    
    total_reward = reward1 + reward2 + reward3 + reward4 + reward5 + reward6
    logger.info(f"=== 测试完成，总奖励: {total_reward} ===")
    
    return {
        "question": question,
        "segment_summaries": segment_summaries,
        "current_summary": current_summary,
        "final_answer": final_answer,
        "total_reward": total_reward
    }


if __name__ == "__main__":
    result = asyncio.run(test_pipeline())
    print(f"测试成功！总奖励: {result['total_reward']}")
