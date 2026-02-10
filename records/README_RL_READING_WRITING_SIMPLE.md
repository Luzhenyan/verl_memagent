# 简化版：基于VERL和SWE的读写能力RL训练

## 1. 简化目标

### 1.1 核心功能
- **读文件**：模型能够读取指定文件内容
- **写总结**：模型能够生成文件内容的总结
- **简单奖励**：基于总结质量给予奖励

### 1.2 训练流程
```
文件 → 读取 → 生成总结 → 评估质量 → 给予奖励
```

## 2. 简化设计

### 2.1 状态空间（简化版）
```python
class SimpleState:
    def __init__(self):
        self.file_content = ""      # 文件内容
        self.generated_summary = "" # 生成的总结
        self.file_path = ""         # 文件路径
```

### 2.2 动作空间（简化版）
```python
class SimpleActions:
    READ_FILE = "read_file"           # 读取文件
    WRITE_SUMMARY = "write_summary"   # 写总结
    FINISH = "finish"                 # 完成任务
```

### 2.3 简单奖励函数
```python
class SimpleReward:
    def calculate_reward(self, state, action, next_state):
        reward = 0
        
        if action == "write_summary":
            # 基于总结质量给予奖励
            summary_quality = self.evaluate_summary_quality(
                next_state.generated_summary,
                state.file_content
            )
            reward = summary_quality * 10  # 简单线性奖励
        
        elif action == "finish":
            # 完成任务给予额外奖励
            reward = 5
        
        return reward
    
    def evaluate_summary_quality(self, summary, original_content):
        """简单的总结质量评估"""
        # 使用简单的指标：总结长度、关键词覆盖率
        if len(summary) < 10:
            return 0.0  # 总结太短
        
        # 简单的关键词匹配
        key_words = self.extract_key_words(original_content)
        matched_words = sum(1 for word in key_words if word.lower() in summary.lower())
        coverage = matched_words / len(key_words) if key_words else 0
        
        return min(coverage, 1.0)  # 归一化到0-1
```

## 3. 环境实现

### 3.1 简化环境类
```python
class SimpleReadingWritingEnv:
    def __init__(self):
        self.file_tools = FileTools()
        self.reward_calculator = SimpleReward()
        self.current_state = SimpleState()
    
    def reset(self, file_path):
        """重置环境，设置新文件"""
        self.current_state = SimpleState()
        self.current_state.file_path = file_path
        return self.current_state
    
    def step(self, action):
        """执行动作"""
        next_state = SimpleState()
        next_state.file_path = self.current_state.file_path
        
        if action == "read_file":
            # 读取文件
            next_state.file_content = self.file_tools.read_file(
                self.current_state.file_path
            )
            reward = 0
        
        elif action == "write_summary":
            # 生成总结（这里需要调用模型）
            next_state.file_content = self.current_state.file_content
            next_state.generated_summary = self.generate_summary(
                self.current_state.file_content
            )
            reward = self.reward_calculator.calculate_reward(
                self.current_state, action, next_state
            )
        
        elif action == "finish":
            next_state = self.current_state
            reward = self.reward_calculator.calculate_reward(
                self.current_state, action, next_state
            )
        
        self.current_state = next_state
        done = (action == "finish")
        
        return next_state, reward, done, {}
    
    def generate_summary(self, content):
        """调用模型生成总结"""
        # 这里需要集成模型调用
        prompt = f"请总结以下内容：\n\n{content}\n\n总结："
        # 调用模型生成总结
        return "生成的总结内容"
```

### 3.2 SWE工具集成
```python
class FileTools:
    def read_file(self, file_path):
        """读取文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"读取文件失败: {e}")
            return ""
    
    def write_file(self, file_path, content):
        """写入文件内容"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"写入文件失败: {e}")
            return False
```

## 4. VERL训练配置

### 4.1 简化配置
```yaml
# config/simple_reading_writing.yaml
trainer:
  project_name: "simple_reading_writing"
  experiment_name: "basic_reader"
  default_local_dir: "/user/wangyicheng/checkpoints"
  
  total_epochs: 50
  save_freq: 10
  test_freq: 5
  
  n_gpus_per_node: 1
  nnodes: 1

data:
  train_files: "/user/wangyicheng/data/simple_docs/train.parquet"
  val_files: "/user/wangyicheng/data/simple_docs/val.parquet"
  train_batch_size: 16
  max_prompt_length: 1024
  max_response_length: 512

actor_rollout_ref:
  model:
    path: "Qwen/Qwen2.5-0.5B-Instruct"
  actor:
    optim:
      lr: 1e-6
    ppo_mini_batch_size: 8
    ppo_micro_batch_size_per_gpu: 2
  rollout:
    name: "vllm"
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.6

critic:
  model:
    path: "Qwen/Qwen2.5-0.5B-Instruct"
  optim:
    lr: 1e-5
  ppo_micro_batch_size_per_gpu: 2

algorithm:
  kl_ctrl:
    kl_coef: 0.001
```

## 5. 训练脚本

### 5.1 简化训练脚本
```bash
#!/bin/bash
# run_simple_reading.sh

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 hydra.run.dir=/user/wangyicheng \
 data.train_files=$HOME/data/simple_docs/train.parquet \
 data.val_files=$HOME/data/simple_docs/val.parquet \
 data.train_batch_size=16 \
 data.max_prompt_length=1024 \
 data.max_response_length=512 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=8 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=2 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=console \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=5 \
 trainer.total_epochs=50 \
 trainer.default_local_dir=/user/wangyicheng/checkpoints 2>&1 | tee /user/wangyicheng/simple_reading.log
```

## 6. 数据格式

### 6.1 简化数据格式
```json
{
  "file_path": "/path/to/document.txt",
  "file_content": "文档内容...",
  "expected_summary": "期望的总结",
  "difficulty": "easy"
}
```

### 6.2 示例数据
```json
{
  "file_path": "docs/article1.txt",
  "file_content": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这些任务包括学习、推理、问题解决、感知和语言理解。",
  "expected_summary": "人工智能是计算机科学分支，创建能执行人类智能任务的系统。",
  "difficulty": "easy"
}
```

## 7. 实现步骤

### 7.1 第一步：创建基础环境
1. 实现 `SimpleReadingWritingEnv` 类
2. 实现 `FileTools` 工具类
3. 实现 `SimpleReward` 奖励函数

### 7.2 第二步：准备数据
1. 创建简单的文档数据集
2. 转换为parquet格式
3. 放在指定目录

### 7.3 第三步：开始训练
1. 运行训练脚本
2. 监控训练过程
3. 评估模型效果

## 8. 预期效果

### 8.1 短期目标
- 模型学会读取文件内容
- 能够生成基本的总结
- 理解简单的奖励机制

### 8.2 评估指标
- **总结质量**：与期望总结的相似度
- **任务完成率**：成功完成读写任务的比例
- **奖励学习**：模型是否学会优化奖励

## 9. 后续扩展

一旦基础版本工作正常，可以逐步添加：
1. 更复杂的奖励函数
2. 重读机制
3. 多文档处理
4. 交互式操作

这个简化版本专注于核心功能，让模型先学会基本的读写能力，为后续的复杂功能打下基础。
