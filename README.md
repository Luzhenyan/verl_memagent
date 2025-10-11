# VERL 分段阅读任务 - 强化学习训练项目

基于 VERL 框架实现的分段阅读理解任务，使用 GRPO 算法训练大语言模型通过工具调用完成长文档阅读理解。

## 📋 目录

- [项目概述](#项目概述)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [使用方式](#使用方式)
- [实现说明](#实现说明)

---

## 🎯 项目概述

本项目实现了一个基于强化学习的分段阅读系统，主要特点：

- **任务**: 长文档阅读理解（TriviaQA 数据集）
- **算法**: GRPO
- **框架**: VERL
- **模型**: Qwen2.5-1.5B-Instruct
- **特色**: 工具调用 + 多轮对话 + 分段阅读策略

---

## 💻 环境要求

### 软件环境
参考verl需要的环境

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/Luzhenyan/verl_memagent.git
cd verl_memagent

# 安装 VERL
pip install -e .

# 安装额外依赖
pip install tensorboard datasets pandas pyarrow
```

---

## 📚 使用方式

### 步骤1：下载模型

运行项目提供的下载脚本：

```bash
cd /home/luzhenyan/verl


# 运行下载脚本
python download_model.py
```

脚本会自动：
1. 下载 Qwen2.5-1.5B-Instruct 模型
2. 保存到 `~/models/Qwen/Qwen2.5-1.5B-Instruct`
3. 测试模型是否正常工作

**模型保存路径**: `~/models/Qwen/Qwen2.5-1.5B-Instruct`

> 注意：首次下载约需 3GB 存储空间，下载时间取决于网络速度

---

### 步骤2：准备数据集

#### 2.1 数据集下载和预处理

运行数据预处理脚本：

```bash
cd /home/luzhenyan/verl

# 安装依赖（如果未安装）
pip install datasets pandas pyarrow

# 运行预处理脚本
python scripts/prepare_full_triviaqa.py
```

脚本会自动：
1. 从 Hugging Face 下载 TriviaQA 数据集（rc.wikipedia 配置）
2. 将长文档分段处理（每段最多 2048 字符，在句子边界截断）
3. 生成 VERL 格式的训练数据（train.parquet 和 val.parquet）
4. 创建文档 JSON 文件和空的摘要文件

#### 2.2 生成的文件结构

```
~/data/triviaqa_docs/
├── train.parquet          # 训练集（1000个样本）
├── val.parquet            # 验证集（100个样本）
├── document_0.json        # 文档文件（包含分段信息）
├── document_1.json
├── ...
└── document_999.json

/user/luzhenyan/           # 可写目录，存放摘要文件
├── document_0_summary.txt  # 空的摘要文件
├── document_1_summary.txt
├── ...
└── document_999_summary.txt
```

#### 2.3 数据格式详解

**Parquet 文件中每个样本**：
```json
{
    "data_source": "segmented_reading",
    "prompt": [
        {
            "role": "user",
            "content": "Please read the document and answer the question: Where in England was Dame Judi Dench born?\n\nDocument information:\n- Document file: /home/luzhenyan/data/triviaqa_docs/document_0.json\n- Total segments: 60\n- Summary file: /user/luzhenyan/document_0_summary.txt\n\nInstructions:\n1. Use read_segment_file to read specific segments from the document\n2. After each reading, use write_summary_file to save your progress\n3. Use read_summary_file to check your previous progress if needed\n\nPlease start reading the document segment by segment."
        }
    ],
    "ability": "reading_comprehension",
    "reward_model": {
        "style": "rule",
        "ground_truth": "York"
    },
    "agent_name": "tool_agent",
    "extra_info": {
        "question": "Where in England was Dame Judi Dench born?",
        "document_file": "/home/luzhenyan/data/triviaqa_docs/document_0.json",
        "num_segments": 60,
        "split": "train"
    }
}
```

**文档文件（document_*.json）**：
```json
{
    "question": "Where in England was Dame Judi Dench born?",
    "segments": [
        {
            "title": "段落1",
            "content": "England is a country that is part of the United Kingdom...",
            "index": 0
        },
        {
            "title": "段落2",
            "content": "With a population of 53 million...",
            "index": 1
        }
    ],
    "num_segments": 60
}
```

---

### 步骤3：启动训练

#### 3.1 配置 TensorBoard

```bash
# 创建日志目录
mkdir -p /data/tensorboard

# 启动 TensorBoard
./start_tensorboard.sh

# 在浏览器访问: http://localhost:6006
```

#### 3.2 运行训练脚本

```bash
cd /home/luzhenyan/verl
./run_segmented_reading_gsm8k_based.sh
```

**训练脚本**: `run_segmented_reading_gsm8k_based.sh`

---

## ⚙️ 关键配置说明

### 训练脚本配置详解

打开 `run_segmented_reading_gsm8k_based.sh`，主要配置项：

#### 1. 环境变量
```bash
export VLLM_USE_V1=1                      # 使用 vLLM v1
export CUDA_VISIBLE_DEVICES=0,1           # 使用的 GPU 编号
export TENSORBOARD_DIR=/data/tensorboard  # TensorBoard 日志目录
```

#### 2. 模型配置
```bash
actor_rollout_ref.model.path=~/models/Qwen/Qwen2.5-1.5B-Instruct  # 模型路径
actor_rollout_ref.model.enable_gradient_checkpointing=True        # 梯度检查点（节省显存）
actor_rollout_ref.model.use_remove_padding=True                   # 优化计算效率
```

#### 3. 数据配置
```bash
data.train_batch_size=2                   # 训练批次大小
data.max_prompt_length=2048               # 最大提示长度
data.max_response_length=2048             # 最大响应长度
data.train_files=/home/luzhenyan/data/triviaqa_docs/train.parquet  # 训练数据
data.val_files=/home/luzhenyan/data/triviaqa_docs/val.parquet      # 验证数据
```

#### 4. GRPO 算法配置
```bash
algorithm.adv_estimator=grpo              # 使用 GRPO 算法
algorithm.use_kl_in_reward=False          # 不在奖励中使用 KL
actor_rollout_ref.rollout.n=8             # 每个 prompt 生成 8 个响应（组采样数量）
trainer.critic_warmup=0                   # Critic 预热步数为 0（GRPO 不需要 Critic）
```

#### 5. 优化器配置
```bash
actor_rollout_ref.actor.optim.lr=1e-6     # 学习率
actor_rollout_ref.actor.use_kl_loss=True  # 启用 KL 损失
actor_rollout_ref.actor.kl_loss_coef=0.001  # KL 损失系数
actor_rollout_ref.actor.kl_loss_type=low_var_kl  # KL 损失类型
```

#### 6. 内存优化配置
```bash
actor_rollout_ref.actor.fsdp_config.param_offload=True      # 参数卸载到 CPU
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True  # 优化器卸载到 CPU
actor_rollout_ref.rollout.gpu_memory_utilization=0.5        # GPU 显存利用率
```

#### 7. 训练配置
```bash
trainer.total_training_steps=1000         # 总训练步数
trainer.total_epochs=3                    # 训练轮数
trainer.test_freq=1                       # 验证频率
trainer.save_freq=-1                      # 保存频率（-1 表示不保存）
trainer.logger="['console','tensorboard']"  # 日志记录器
```

#### 8. 工具配置
```bash
actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/verl/utils/tools/segmented_reading_tools.yaml"
```

---

## 🔧 实现说明

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        训练流程                              │
├─────────────────────────────────────────────────────────────┤
│  1. 数据加载 (DataLoader)                                    │
│     └─> TriviaQA Dataset (Parquet)                          │
│                                                              │
│  2. 模型推理 (Rollout)                                       │
│     └─> vLLM + Tool Calling                                 │
│         ├─> read_segment_file()                             │
│         ├─> write_summary_file()                            │
│         └─> read_summary_file()                             │
│                                                              │
│  3. 奖励计算 (Reward)                                        │
│     └─> 基于答案准确性的规则奖励                             │
│                                                              │
│  4. 优势估计 (Advantage)                                     │
│     └─> GRPO: 组内相对优势计算                              │
│                                                              │
│  5. 策略更新 (Actor Update)                                  │
│                                      │
└─────────────────────────────────────────────────────────────┘
```

### 核心组件

#### 1. 工具系统

**文件**: `verl/utils/tools/segmented_reading_tools.yaml`

定义了三个工具函数：

```yaml
- read_segment_file: 读取指定文档片段
- write_summary_file: 保存阅读摘要
- read_summary_file: 读取之前的摘要
```

**实现**: `verl/tools/reading_tools.py`

#### 2. Agent Loop

**文件**: `verl/experimental/agent_loop/tool_agent_loop.py`

负责：
- 解析模型输出的工具调用
- 执行工具函数
- 管理多轮对话
- 控制最大轮次（max_assistant_turns=5）

#### 3. 奖励函数

**文件**: `verl/utils/reward_score/segmented_reading.py`

实现 `compute_score()` 函数：
```python
def compute_score(prompts, responses, reward_models):
    # 提取答案
    # 与 ground_truth 比较
    # 返回奖励分数 (0.0 - 1.0)
```

**奖励策略**:
- 答案正确: 1.0 ✅
- 答案错误或未找到答案: 0.0 ❌

多层次匹配策略判断答案是否正确：
1. 精确匹配（规范化后完全相同）
2. 包含匹配（短答案 ≤ 20 字符）
3. 数字匹配（提取数字精确比较）
4. 关键词匹配（至少 60% 词汇重叠）



### 修改的核心文件

#### 新增文件

| 文件路径 | 说明 |
|---------|------|
| `verl/tools/reading_tools.py` | 分段阅读工具实现 |
| `verl/utils/tools/segmented_reading_tools.yaml` | 工具配置文件 |
| `verl/utils/reward_score/segmented_reading.py` | 奖励函数 |
| `scripts/prepare_full_triviaqa.py` | 数据预处理脚本 |
| `run_segmented_reading_gsm8k_based.sh` | 训练启动脚本 |
| `start_tensorboard.sh` | TensorBoard 启动脚本 |

#### 修改文件

| 文件路径 | 修改内容 |
|---------|---------|
| `verl/utils/reward_score/__init__.py` | 注册 segmented_reading 奖励函数 |
| `verl/experimental/agent_loop/tool_agent_loop.py` | 增强工具调用解析 |
| `verl/trainer/ppo/ray_trainer.py` | 集成自定义奖励函数 |

---

