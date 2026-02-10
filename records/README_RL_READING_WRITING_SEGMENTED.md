# 分段阅读总结RL训练方案

## 1. 核心概念

### 1.1 训练目标
让模型学会：
- **分段阅读**：一段一段地读取文档内容
- **分段总结**：为每段内容生成总结
- **智能决策**：基于问题和已有总结决定下一步读什么
- **逐步推理**：通过多段阅读和总结最终回答问题

### 1.2 训练流程
```
问题 → 读第1段 → 写总结1 → 更新当前总结 → 读第2段 → 写总结2 → 更新当前总结 → ... → 生成最终答案
```

## 2. 详细设计

### 2.1 状态空间
```python
class SegmentedState:
    def __init__(self):
        self.question = ""                    # 待回答问题
        self.document_segments = []           # 文档分段列表
        self.current_segment_index = 0        # 当前阅读的段落索引
        self.read_segments = []               # 已读段落索引列表
        self.segment_summaries = {}           # 每段的总结 {index: summary}
        self.current_summary = ""             # 基于前面所有总结的当前总结
        self.final_answer = ""                # 最终答案
        self.reading_history = []             # 阅读历史
```

### 2.2 动作空间
```python
class SegmentedActions:
    # 基础动作
    READ_NEXT_SEGMENT = "read_next_segment"      # 读下一段
    READ_PREVIOUS_SEGMENT = "read_previous_segment"  # 重读前一段
    READ_SPECIFIC_SEGMENT = "read_specific_segment"  # 读指定段落
    WRITE_SEGMENT_SUMMARY = "write_segment_summary"  # 写段落总结
    UPDATE_CURRENT_SUMMARY = "update_current_summary"  # 更新当前总结
    GENERATE_FINAL_ANSWER = "generate_final_answer"  # 生成最终答案
    FINISH = "finish"                           # 完成任务
```

### 2.3 奖励函数设计
```python
class SegmentedReward:
    def calculate_reward(self, state, action, next_state):
        reward = 0
        
        if action == "read_specific_segment":
            # 只有选择相关段落才给奖励
            relevance = self.evaluate_segment_relevance(
                state.document_segments[state.current_segment_index],
                state.question
            )
            reward += relevance * 5
        
        elif action == "write_segment_summary":
            # 只有总结质量好才给奖励
            summary_quality = self.evaluate_summary_quality(
                next_state.segment_summaries[state.current_segment_index],
                state.document_segments[state.current_segment_index]
            )
            if summary_quality > 0.5:  # 只有质量超过阈值才给奖励
                reward += summary_quality * 3
        
        elif action == "update_current_summary":
            # 只有当前总结对回答问题有帮助才给奖励
            helpfulness = self.evaluate_summary_helpfulness(
                next_state.current_summary,
                state.question
            )
            reward += helpfulness * 5
        
        elif action == "generate_final_answer":
            # 只有答案准确才给奖励
            answer_accuracy = self.evaluate_answer_accuracy(
                next_state.final_answer,
                state.question
            )
            reward += answer_accuracy * 10
        
        # 其他动作（如read_next_segment, finish等）不给奖励
        

        
        return reward
    
    def evaluate_segment_relevance(self, segment, question):
        """评估段落与问题的相关性"""
        # 简单的关键词匹配
        question_words = set(question.lower().split())
        segment_words = set(segment.lower().split())
        overlap = len(question_words & segment_words)
        return min(overlap / len(question_words), 1.0) if question_words else 0.0
    
    def evaluate_summary_quality(self, summary, original_segment):
        """评估总结质量"""
        if len(summary) < 10:
            return 0.0
        # 检查总结是否包含原文的关键信息
        key_words = self.extract_key_words(original_segment)
        matched = sum(1 for word in key_words if word.lower() in summary.lower())
        return matched / len(key_words) if key_words else 0.0
    
    def evaluate_summary_helpfulness(self, summary, question):
        """评估总结对回答问题的帮助程度"""
        if len(summary) < 10:
            return 0.0
        # 检查总结是否包含回答问题所需的信息
        question_words = set(question.lower().split())
        summary_words = set(summary.lower().split())
        relevance = len(question_words & summary_words) / len(question_words) if question_words else 0.0
        return min(relevance, 1.0)
    
    def evaluate_answer_accuracy(self, answer, question):
        """评估答案准确性"""
        if len(answer) < 10:
            return 0.0
        # 简单的答案质量评估（可以后续改进）
        return min(len(answer) / 50, 1.0)
```

## 3. 环境实现

### 3.1 分段阅读环境
```python
class SegmentedReadingEnv:
    def __init__(self):
        self.file_tools = FileTools()
        self.reward_calculator = SegmentedReward()
        self.current_state = SegmentedState()
    
    def reset(self, file_path, question):
        """重置环境"""
        self.current_state = SegmentedState()
        self.current_state.question = question
        
        # 读取文档并分段
        content = self.file_tools.read_file(file_path)
        self.current_state.document_segments = self.segment_document(content)
        
        return self.current_state
    
    def step(self, action, **kwargs):
        """执行动作"""
        next_state = copy.deepcopy(self.current_state)
        
        if action == "read_next_segment":
            if next_state.current_segment_index < len(next_state.document_segments) - 1:
                next_state.current_segment_index += 1
                if next_state.current_segment_index not in next_state.read_segments:
                    next_state.read_segments.append(next_state.current_segment_index)
            reward = 0
        
        elif action == "read_previous_segment":
            if next_state.current_segment_index > 0:
                next_state.current_segment_index -= 1
            reward = 0
        
        elif action == "read_specific_segment":
            segment_index = kwargs.get('segment_index', 0)
            if 0 <= segment_index < len(next_state.document_segments):
                next_state.current_segment_index = segment_index
                if segment_index not in next_state.read_segments:
                    next_state.read_segments.append(segment_index)
            reward = self.reward_calculator.calculate_reward(
                self.current_state, action, next_state
            )
        
        elif action == "write_segment_summary":
            current_segment = next_state.document_segments[next_state.current_segment_index]
            summary = self.generate_segment_summary(current_segment)
            next_state.segment_summaries[next_state.current_segment_index] = summary
            reward = self.reward_calculator.calculate_reward(
                self.current_state, action, next_state
            )
        
        elif action == "update_current_summary":
            current_summary = self.generate_current_summary(
                next_state.segment_summaries,
                next_state.question
            )
            next_state.current_summary = current_summary
            reward = self.reward_calculator.calculate_reward(
                self.current_state, action, next_state
            )
        
        elif action == "generate_final_answer":
            final_answer = self.generate_final_answer(
                next_state.current_summary,
                next_state.question
            )
            next_state.final_answer = final_answer
            reward = self.reward_calculator.calculate_reward(
                self.current_state, action, next_state
            )
        
        elif action == "finish":
            reward = self.reward_calculator.calculate_reward(
                self.current_state, action, next_state
            )
        
        self.current_state = next_state
        done = (action == "finish")
        
        return next_state, reward, done, {}
    
    def segment_document(self, content, max_length=500):
        """将文档分割为段落"""
        # 简单的按句号分割，可以根据需要优化
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
    
    def generate_segment_summary(self, segment_content):
        """生成段落总结"""
        prompt = f"请总结以下段落的关键信息：\n\n{segment_content}\n\n总结："
        # 调用模型生成总结
        return "段落总结内容"
    
    def generate_current_summary(self, segment_summaries, question):
        """生成当前总结"""
        summaries_text = "\n".join([f"第{i+1}段：{summary}" for i, summary in segment_summaries.items()])
        prompt = f"基于以下段落总结和问题，生成一个当前总结：\n\n问题：{question}\n\n段落总结：\n{summaries_text}\n\n当前总结："
        # 调用模型生成当前总结
        return "当前总结内容"
    
    def generate_final_answer(self, current_summary, question):
        """生成最终答案"""
        prompt = f"基于以下当前总结回答问题：\n\n问题：{question}\n\n当前总结：{current_summary}\n\n答案："
        # 调用模型生成答案
        return "最终答案内容"
```

## 4. 训练配置

### 4.1 分段阅读训练配置
```yaml
# config/segmented_reading.yaml
trainer:
  project_name: "segmented_reading"
  experiment_name: "smart_reader"
  default_local_dir: "/user/wangyicheng/checkpoints"
  
  total_epochs: 100
  save_freq: 10
  test_freq: 5
  
  n_gpus_per_node: 1
  nnodes: 1

data:
  train_files: "/user/wangyicheng/data/segmented_docs/train.parquet"
  val_files: "/user/wangyicheng/data/segmented_docs/val.parquet"
  train_batch_size: 16
  max_prompt_length: 2048
  max_response_length: 1024

actor_rollout_ref:
  model:
    path: "Qwen/Qwen2.5-7B-Instruct"
  actor:
    optim:
      lr: 1e-6
    ppo_mini_batch_size: 8
    ppo_micro_batch_size_per_gpu: 4
  rollout:
    name: "vllm"
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.8

critic:
  model:
    path: "Qwen/Qwen2.5-7B-Instruct"
  optim:
    lr: 1e-5
  ppo_micro_batch_size_per_gpu: 4

algorithm:
  kl_ctrl:
    kl_coef: 0.001

# 分段阅读特定配置
segmented_reading:
  max_segment_length: 500
  max_segments_per_doc: 20
  min_segments_to_read: 2
  max_segments_to_read: 15
```

## 5. 数据格式

### 5.1 分段数据格式
```json
{
  "file_path": "/path/to/document.txt",
  "question": "文档中提到了哪些主要观点？",
  "segments": [
    "第一段内容...",
    "第二段内容...",
    "第三段内容..."
  ],
  "segment_summaries": [
    "第一段总结...",
    "第二段总结...",
    "第三段总结..."
  ],
  "current_summary": "基于前面所有段落的当前总结",
  "expected_answer": "期望的最终答案",
  "relevant_segments": [0, 2, 4],  // 回答问题相关的段落索引
  "difficulty": "medium"
}
```

### 5.2 示例数据
```json
{
  "file_path": "docs/ai_article.txt",
  "question": "人工智能的发展历程是怎样的？",
  "segments": [
    "人工智能的概念最早由艾伦·图灵在1950年提出。他提出了著名的图灵测试，用于判断机器是否具有智能。",
    "在20世纪60年代，人工智能研究主要集中在符号推理和专家系统上。这一时期出现了许多重要的AI程序。",
    "到了80年代，机器学习开始兴起。神经网络和深度学习的概念逐渐发展起来。",
    "21世纪以来，随着大数据和计算能力的提升，深度学习取得了突破性进展。"
  ],
  "segment_summaries": [
    "图灵提出AI概念和图灵测试",
    "60年代发展符号推理和专家系统",
    "80年代兴起机器学习和神经网络",
    "21世纪深度学习取得突破"
  ],
  "current_summary": "AI从图灵概念提出开始，经历了符号推理、专家系统、机器学习到深度学习的发展历程",
  "expected_answer": "人工智能从1950年图灵提出概念开始，经历了符号推理、专家系统、机器学习、深度学习等发展阶段。",
  "relevant_segments": [0, 1, 2, 3],
  "difficulty": "medium"
}
```

## 6. 训练脚本

### 6.1 分段阅读训练脚本
```bash
#!/bin/bash
# run_segmented_reading.sh

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 hydra.run.dir=/user/wangyicheng \
 data.train_files=$HOME/data/segmented_docs/train.parquet \
 data.val_files=$HOME/data/segmented_docs/val.parquet \
 data.train_batch_size=16 \
 data.max_prompt_length=2048 \
 data.max_response_length=1024 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=8 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-7B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=console \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=5 \
 trainer.total_epochs=100 \
 trainer.default_local_dir=/user/wangyicheng/checkpoints 2>&1 | tee /user/wangyicheng/segmented_reading.log
```

## 7. 预期效果

### 7.1 模型学会的能力
1. **智能分段阅读**：根据问题和已有总结选择最相关的段落
2. **渐进式总结**：为每段生成总结，并基于前面所有总结生成新总结
3. **上下文感知**：基于已有信息决定下一步行动
4. **高效阅读**：避免阅读无关段落，提高效率

### 7.2 评估指标
- **阅读效率**：阅读段落数量与答案质量的关系
- **总结质量**：段落总结和当前总结的质量
- **答案准确性**：最终答案的准确性
- **策略合理性**：阅读顺序和段落选择的合理性

## 8. 实现优势

### 8.1 相比整体阅读的优势
1. **处理长文档**：可以处理超出模型上下文长度的文档
2. **精确聚焦**：只阅读回答问题相关的段落
3. **渐进推理**：通过多步推理提高答案质量
4. **可解释性**：可以追踪模型的阅读和推理过程

### 8.2 训练优势
1. **密集奖励**：每个动作都有相应的奖励信号
2. **策略学习**：模型学会何时读、读什么、如何总结
3. **适应性**：可以适应不同类型的文档和问题

这个方案让模型学会像人类一样分段阅读和思考，通过多步推理来解决复杂的长文档理解任务。
