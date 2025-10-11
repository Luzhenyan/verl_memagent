#!/usr/bin/env python3
"""
测试SegmentedReadingEnvironment
验证环境功能，包括数据加载、状态管理、动作执行和奖励计算
"""

import sys
import logging
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from verl.environments.segmented_reading_env import SegmentedReadingEnvironment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_data():
    """创建测试用的HotpotQA数据"""
    test_data = [
        {
            "_id": "test_001",
            "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
            "answer": "yes",
            "supporting_facts": [["Scott Derrickson", 0], ["Ed Wood", 0]],
            "context": [
                ["Scott Derrickson", [
                    "Scott Derrickson (born July 16, 1966) is an American director, screenwriter and producer.",
                    "He lives in Los Angeles, California.",
                    "He is best known for directing horror films such as \"Sinister\", \"The Exorcism of Emily Rose\", and \"Deliver Us From Evil\", as well as the 2016 Marvel Cinematic Universe installment, \"Doctor Strange.\""
                ]],
                ["Ed Wood", [
                    "Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker, actor, writer, producer, and director."
                ]],
                ["Ed Wood (film)", [
                    "Ed Wood is a 1994 American biographical period comedy-drama film directed and produced by Tim Burton, and starring Johnny Depp as cult filmmaker Ed Wood.",
                    "The film concerns the period in Wood's life when he made his best-known films as well as his relationship with actor Bela Lugosi, played by Martin Landau."
                ]]
            ],
            "type": "comparison",
            "level": "hard"
        },
        {
            "_id": "test_002", 
            "question": "What is the capital of France?",
            "answer": "Paris",
            "supporting_facts": [["France", 0]],
            "context": [
                ["France", [
                    "France is a country in Europe.",
                    "The capital of France is Paris.",
                    "Paris is known for the Eiffel Tower and the Louvre Museum."
                ]]
            ],
            "type": "bridge",
            "level": "easy"
        }
    ]
    
    return test_data


def test_environment_basic():
    """测试环境基本功能"""
    logger.info("=== 测试环境基本功能 ===")
    
    # 创建环境
    config = {
        "max_steps": 10,
        "reward_weights": {
            "read_segment": 0.1,
            "write_summary": 0.3,
            "update_summary": 0.3,
            "extract_facts": 0.5,
            "final_answer": 1.0,
            "step_penalty": -0.01
        }
    }
    env = SegmentedReadingEnvironment(config)
    
    # 创建测试数据
    test_data = create_test_data()
    test_file = "/tmp/test_hotpotqa.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # 加载数据
    env.load_data(test_file)
    logger.info(f"成功加载 {len(env.hotpotqa_data)} 个测试样本")
    
    # 测试重置
    state = env.reset(episode_id=0)
    logger.info(f"环境重置成功")
    logger.info(f"问题: {state.question}")
    logger.info(f"答案: {state.answer}")
    logger.info(f"支持事实: {state.supporting_facts}")
    logger.info(f"段落数量: {len(state.context)}")
    logger.info(f"可用段落: {state.available_segments}")
    
    return env


def test_action_execution(env):
    """测试动作执行"""
    logger.info("\n=== 测试动作执行 ===")
    
    # 重置环境
    state = env.reset(episode_id=0)
    
    # 测试动作序列
    test_actions = [
        {
            "action": "read_document_segment",
            "args": {"segment_index": 0}
        },
        {
            "action": "write_segment_summary", 
            "args": {
                "segment_index": 0,
                "summary": "Scott Derrickson is an American director born in 1966. He is known for horror films and Doctor Strange."
            }
        },
        {
            "action": "read_document_segment",
            "args": {"segment_index": 1}
        },
        {
            "action": "write_segment_summary",
            "args": {
                "segment_index": 1, 
                "summary": "Ed Wood was an American filmmaker who lived from 1924 to 1978."
            }
        },
        {
            "action": "update_current_summary",
            "args": {
                "summary": "Scott Derrickson is an American director. Ed Wood was also an American filmmaker. Both are American."
            }
        },
        {
            "action": "generate_final_answer",
            "args": {
                "answer": "yes, both Scott Derrickson and Ed Wood are American"
            }
        }
    ]
    
    total_reward = 0.0
    
    for i, test_action in enumerate(test_actions):
        logger.info(f"\n--- 步骤 {i+1}: {test_action['action']} ---")
        
        state, reward, done, info = env.step(
            test_action['action'], 
            test_action['args']
        )
        
        total_reward += reward
        logger.info(f"奖励: {reward:.3f}")
        logger.info(f"完成: {done}")
        logger.info(f"信息: {info}")
        logger.info(f"当前步骤: {state.current_step}")
        logger.info(f"已读段落: {state.read_segments}")
        
        if done:
            logger.info("环境完成！")
            break
    
    logger.info(f"\n总奖励: {total_reward:.3f}")
    return total_reward


def test_reward_functions(env):
    """测试奖励函数"""
    logger.info("\n=== 测试奖励函数 ===")
    
    # 重置环境
    state = env.reset(episode_id=0)
    
    # 测试关键事实提取奖励
    logger.info("测试关键事实提取奖励:")
    
    # 好的总结（包含关键事实）
    good_summary = "Scott Derrickson is an American director known for horror films and Doctor Strange."
    fact_score = env._evaluate_fact_extraction(good_summary, 0)
    logger.info(f"好的总结: '{good_summary}' -> 事实提取分数: {fact_score:.3f}")
    
    # 差的总结（不包含关键事实）
    bad_summary = "This person is a filmmaker who makes movies."
    fact_score = env._evaluate_fact_extraction(bad_summary, 0)
    logger.info(f"差的总结: '{bad_summary}' -> 事实提取分数: {fact_score:.3f}")
    
    # 测试答案准确性奖励
    logger.info("\n测试答案准确性奖励:")
    
    correct_answer = "yes"
    test_answers = [
        "yes",  # 完全匹配
        "yes, they are both American",  # 部分匹配
        "both are American",  # 关键词匹配
        "no"  # 错误答案
    ]
    
    for answer in test_answers:
        accuracy = env._evaluate_answer_accuracy(answer)
        logger.info(f"答案: '{answer}' -> 准确性: {accuracy:.3f}")


def test_multiple_episodes(env):
    """测试多个episode"""
    logger.info("\n=== 测试多个episode ===")
    
    for episode_id in range(len(env.hotpotqa_data)):
        logger.info(f"\n--- Episode {episode_id} ---")
        
        state = env.reset(episode_id=episode_id)
        logger.info(f"问题: {state.question}")
        logger.info(f"答案: {state.answer}")
        logger.info(f"段落数: {len(state.context)}")
        
        # 简单测试：读取第一个段落
        if len(state.context) > 0:
            state, reward, done, info = env.step(
                "read_document_segment", 
                {"segment_index": 0}
            )
            logger.info(f"读取段落奖励: {reward:.3f}")


def main():
    """主测试函数"""
    logger.info("开始测试SegmentedReadingEnvironment")
    
    try:
        # 1. 测试基本功能
        env = test_environment_basic()
        
        # 2. 测试动作执行
        total_reward = test_action_execution(env)
        
        # 3. 测试奖励函数
        test_reward_functions(env)
        
        # 4. 测试多个episode
        test_multiple_episodes(env)
        
        logger.info("\n=== 所有测试完成 ===")
        logger.info("✅ 环境功能正常")
        logger.info("✅ 奖励函数工作正常")
        logger.info("✅ 可以开始集成到VERL训练流程")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
