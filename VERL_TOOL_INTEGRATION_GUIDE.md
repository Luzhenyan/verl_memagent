# VERL工具集成指南 - 分段阅读RL训练

## 概述

本文档说明如何在VERL中正确集成工具系统，实现分段阅读的强化学习训练。基于VERL官方文档，我们需要使用其内置的多轮对话和工具调用系统，而不是自定义环境。

## 当前文件状态分析

### 已存在的文件

#### 1. 工具文件
- `verl/verl/tools/reading_tools.py` - 基础阅读工具（已实现）
- `verl/verl/tools/swe_reading_tools.py` - SWE增强工具（已实现）

#### 2. 环境文件
- `verl/verl/environments/segmented_reading_env.py` - 自定义环境（**不需要使用**）

#### 3. 数据集文件
- `verl/verl/utils/dataset/segmented_reading_dataset.py` - 自定义数据集（**不需要使用**）

#### 4. 配置文件
- `verl/verl/trainer/config/segmented_reading.yaml` - 自定义配置（**需要修改**）

#### 5. 数据准备文件
- `verl/scripts/prepare_triviaqa.py` - 数据准备脚本（已修改为分段格式）

## 正确的VERL工具集成方案

### 方案选择：使用VERL内置工具系统

根据VERL官方文档，我们应该使用：
1. **Multi-turn Rollout Support** - 多轮对话支持
2. **Tool Config Path** - 工具配置文件
3. **Custom Reward Function** - 自定义奖励函数
4. **Interaction System** - 交互系统

**不推荐使用**：
- 自定义环境（SegmentedReadingEnvironment）
- 自定义数据集类（SegmentedReadingDataset）

### 需要的文件结构

```
verl/
├── verl/
│   ├── tools/
│   │   ├── reading_tools.py                    # 工具实现
│   │   └── __init__.py                         # 工具注册
│   ├── utils/
│   │   ├── tools/
│   │   │   └── segmented_reading_tools.yaml    # 工具配置文件
│   │   └── reward_score/
│   │       └── segmented_reading.py            # 奖励函数
│   └── trainer/
│       └── config/
│           └── segmented_reading.yaml          # 训练配置
├── scripts/
│   ├── prepare_triviaqa.py                     # 数据准备
│   └── run_simple_demo.sh                      # 训练脚本
└── data/
    └── triviaqa_docs/
        ├── train_small.parquet                 # 训练数据
        └── val.parquet                         # 验证数据
```

## 文件详细说明

### 1. 工具实现文件

#### `verl/verl/tools/reading_tools.py`
**作用**: 实现分段阅读的核心工具
**内容**:
- `ReadDocumentTool` - 读取文档段落
- `WriteSummaryTool` - 生成段落总结
- `UpdateCurrentSummaryTool` - 更新综合总结
- `GenerateFinalAnswerTool` - 生成最终答案

**状态**: ✅ 已实现，需要确保符合VERL工具接口

#### `verl/verl/tools/__init__.py`
**作用**: 注册工具到VERL系统
**需要添加**:
```python
from .reading_tools import (
    ReadDocumentTool,
    WriteSummaryTool,
    UpdateCurrentSummaryTool,
    GenerateFinalAnswerTool
)

__all__ = [
    "ReadDocumentTool",
    "WriteSummaryTool", 
    "UpdateCurrentSummaryTool",
    "GenerateFinalAnswerTool"
]
```

### 2. 工具配置文件

#### `verl/verl/utils/tools/segmented_reading_tools.yaml`
**作用**: 定义工具配置，供VERL加载
**内容**:
```yaml
tools:
  - class_name: "ReadDocumentTool"
    config:
        type: native
    tool_schema:
        name: "read_document_segment"
        description: "Read a specific segment of a document"
        parameters:
            type: object
            properties:
                segment_index:
                    type: integer
                    description: "Index of the segment to read"
            required: ["segment_index"]
    
  - class_name: "WriteSummaryTool"
    config:
        type: native
    tool_schema:
        name: "write_summary"
        description: "Generate a summary for a document segment"
        parameters:
            type: object
            properties:
                segment_index:
                    type: integer
                    description: "Index of the segment"
                summary:
                    type: string
                    description: "Summary content"
            required: ["segment_index", "summary"]
    
  - class_name: "UpdateCurrentSummaryTool"
    config:
        type: native
    tool_schema:
        name: "update_current_summary"
        description: "Update the current comprehensive summary"
        parameters:
            type: object
            properties:
                summary:
                    type: string
                    description: "Updated summary content"
            required: ["summary"]
    
  - class_name: "GenerateFinalAnswerTool"
    config:
        type: native
    tool_schema:
        name: "generate_final_answer"
        description: "Generate the final answer based on current summary"
        parameters:
            type: object
            properties:
                answer:
                    type: string
                    description: "Final answer"
            required: ["answer"]
```

**状态**: ✅ 已创建

### 3. 奖励函数文件

#### `verl/verl/utils/reward_score/segmented_reading.py`
**作用**: 实现符合VERL接口的奖励函数
**接口要求**:
```python
def compute_segmented_reading_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] = None,
    **kwargs
) -> float:
```

**奖励计算**:
- 工具使用奖励 (30%)
- 分段阅读奖励 (30%)
- 总结质量奖励 (20%)
- 最终答案准确性 (20%)

**状态**: ✅ 已实现

### 4. 训练配置文件

#### `verl/verl/trainer/config/segmented_reading.yaml`
**作用**: 定义训练配置
**需要包含**:
```yaml
# 多轮对话配置
actor_rollout_ref:
  rollout:
    multi_turn:
      enable: true
      max_assistant_turns: 5
      max_user_turns: 5
      format: hermes
    tool_config_path: verl/utils/tools/segmented_reading_tools.yaml

# 自定义奖励函数
custom_reward_function:
  path: verl/utils/reward_score/segmented_reading.py
  name: compute_segmented_reading_reward
```

**状态**: ❌ 需要修改

### 5. 数据准备文件

#### `verl/scripts/prepare_triviaqa.py`
**作用**: 准备分段数据
**功能**:
- 加载TriviaQA数据集
- 按2048字符分段
- 转换为VERL格式
- 保存为Parquet文件

**状态**: ✅ 已实现

### 6. 训练脚本

#### `verl/scripts/run_simple_demo.sh`
**作用**: 运行训练
**配置要点**:
- 启用多轮对话
- 指定工具配置路径
- 指定奖励函数
- 使用分段数据

**状态**: ✅ 已修改

## 实施步骤

### 步骤1: 确保工具注册
```bash
# 检查工具是否正确注册
cd /home/wangyicheng/verl
python -c "from verl.tools import ReadDocumentTool; print('Tools registered successfully')"
```

### 步骤2: 验证工具配置
```bash
# 测试工具配置加载
python -c "import yaml; config = yaml.safe_load(open('verl/verl/utils/tools/segmented_reading_tools.yaml')); print('Tool config loaded:', config)"
```

### 步骤3: 测试奖励函数
```bash
# 测试奖励函数
python -c "from verl.utils.reward_score.segmented_reading import compute_segmented_reading_reward; print('Reward function imported successfully')"
```

### 步骤4: 运行训练
```bash
# 运行训练
cd /home/wangyicheng/verl
./run_simple_demo.sh
```

## 关键配置说明

根据VERL官方文档，使用Tool Agent Loop需要以下关键配置：

### 必需配置
```yaml
# 数据配置
data.return_raw_chat: true

# 异步rollout模式
actor_rollout_ref.rollout.mode: async

# 多轮对话配置
actor_rollout_ref.rollout.multi_turn: True  # 注意：是True而不是enable: true
actor_rollout_ref.rollout.name: "sglang"    # 或"vllm"

# 工具配置
actor_rollout_ref.rollout.tool_kwargs:
    tools_config_file: verl/utils/tools/segmented_reading_tools.yaml
```

### 工具配置
```yaml
actor_rollout_ref.rollout.tool_config_path: verl/utils/tools/segmented_reading_tools.yaml
```

### 奖励函数配置
```yaml
custom_reward_function.path: verl/utils/reward_score/segmented_reading.py
custom_reward_function.name: compute_segmented_reading_reward
```

## 数据格式要求

### 输入数据格式
```json
{
  "prompt": [{"role": "user", "content": "问题内容"}],
  "response": [{"role": "assistant", "content": "回答内容"}],
  "question": "问题",
  "answer": "答案",
  "context": "完整文档",
  "segments": [{"title": "段落1", "content": "内容", "index": 0}],
  "num_segments": 60,
  "agent_name": "tool_agent_loop"  // 关键字段：指定使用工具代理循环
}
```

### 多轮对话格式
```
User: 请阅读文档并回答问题：Where in England was Dame Judi Dench born?

Assistant: 我将分段阅读文档来回答这个问题。

Tool: read_document_segment
Args: {"segment_index": 0}

User: 这是第一段内容：[段落内容]

Assistant: 基于第一段内容，我生成了以下总结：[总结内容]

Tool: write_summary
Args: {"segment_index": 0, "summary": "总结内容"}

...

User: 基于所有阅读的内容，请给出最终答案。

Assistant: 最终答案是：York
```

## 常见问题解决

### 1. 工具未找到
**错误**: `ModuleNotFoundError: No module named 'verl.tools'`
**解决**: 确保工具在`verl/verl/tools/__init__.py`中正确注册

### 2. 工具配置加载失败
**错误**: `ConfigAttributeError: Key 'tool_config_path' is not in struct`
**解决**: 检查配置文件中的路径是否正确

### 3. 奖励函数未找到
**错误**: `AttributeError: module has no attribute 'compute_segmented_reading_reward'`
**解决**: 确保奖励函数名称正确，文件路径正确

### 4. 多轮对话格式错误
**错误**: 对话格式不符合要求
**解决**: 确保使用hermes格式，工具调用格式正确

## 总结

正确的VERL工具集成方案是：
1. **使用VERL内置的多轮对话系统**，而不是自定义环境
2. **通过工具配置文件注册工具**，而不是在代码中直接实例化
3. **使用自定义奖励函数**来评估工具使用和任务完成质量
4. **遵循VERL的数据格式和接口规范**

这样可以充分利用VERL的现有功能，避免重复造轮子，同时确保系统的稳定性和可扩展性。

## 当前需要执行的具体步骤

### 立即需要做的修改

#### 1. 修改工具注册文件
**文件**: `verl/verl/tools/__init__.py`
**操作**: 添加工具导入和注册
```python
from .reading_tools import (
    ReadDocumentTool,
    WriteSummaryTool,
    UpdateCurrentSummaryTool,
    GenerateFinalAnswerTool
)

__all__ = [
    "ReadDocumentTool",
    "WriteSummaryTool", 
    "UpdateCurrentSummaryTool",
    "GenerateFinalAnswerTool"
]
```

#### 2. 修改训练配置文件
**文件**: `verl/verl/trainer/config/segmented_reading.yaml`
**操作**: 添加多轮对话和工具配置
```yaml
# 在现有配置基础上添加
data:
  return_raw_chat: true  # 必需：返回原始对话

actor_rollout_ref:
  rollout:
    mode: async  # 必需：异步模式
    multi_turn: True  # 注意：是True而不是enable: true
    name: "sglang"    # 或"vllm"
    tool_kwargs:
      tools_config_file: verl/utils/tools/segmented_reading_tools.yaml

custom_reward_function:
  path: verl/utils/reward_score/segmented_reading.py
  name: compute_segmented_reading_reward
```

#### 3. 修改数据准备脚本
**文件**: `verl/scripts/prepare_triviaqa.py`
**操作**: 添加agent_name字段
```python
# 在创建VERL格式数据时添加agent_name字段
verl_sample = {
    "prompt": prompt,
    "response": response,
    "question": sample["question"],
    "answer": sample['answer']['value'],
    "context": combined_context,
    "segments": segments,
    "num_segments": len(segments),
    "agent_name": "tool_agent_loop",  # 关键字段：指定使用工具代理循环
    "extra_info": {
        "index": i,
        "tools_kwargs": {"dummy": "value"},
        "interaction_kwargs": {"dummy": "value"},
        "need_tools_kwargs": False
    }
}
```

#### 4. 验证工具配置
**操作**: 测试工具配置是否正确加载
```bash
cd /home/wangyicheng/verl
python -c "import yaml; config = yaml.safe_load(open('verl/verl/utils/tools/segmented_reading_tools.yaml')); print('Tool config loaded successfully')"
```

#### 4. 测试工具注册
**操作**: 验证工具是否正确注册
```bash
cd /home/wangyicheng/verl
python -c "from verl.tools import ReadDocumentTool; print('Tools registered successfully')"
```

#### 5. 测试奖励函数
**操作**: 验证奖励函数是否可以导入
```bash
cd /home/wangyicheng/verl
python -c "from verl.utils.reward_score.segmented_reading import compute_segmented_reading_reward; print('Reward function imported successfully')"
```

### 执行顺序

1. **第一步**: 修改工具注册文件
2. **第二步**: 修改训练配置文件  
3. **第三步**: 修改数据准备脚本（添加agent_name字段）
4. **第四步**: 重新生成数据
5. **第五步**: 验证工具配置加载
6. **第六步**: 测试工具注册
7. **第七步**: 测试奖励函数
8. **第八步**: 运行训练脚本

### 预期结果

- 工具配置正确加载
- 工具成功注册到VERL系统
- 奖励函数可以正常导入
- 训练脚本可以启动多轮对话和工具调用

### 可能遇到的问题

1. **工具注册失败**: 检查`__init__.py`文件格式
2. **配置加载失败**: 检查YAML文件语法和路径
3. **模块导入失败**: 检查Python路径和文件位置
4. **训练启动失败**: 检查配置参数是否正确

### 成功标准

- 所有验证命令都成功执行
- 训练脚本可以启动
- 模型能够进行多轮对话
- 工具能够被正确调用
- 奖励函数能够正常计算奖励

### 重要注意事项

1. **工具调用错误**: 根据文档，模型可能有时会生成不正确的toolcall标签，会出现"Failed to decode tool call"错误，但这不影响训练。

2. **Tokenization检查**: VERL会进行tokenization sanity check，如果看到警告，可以配置`tokenization_sanity_check_mode`参数。

3. **多模态输入**: 如果工具返回多模态输入，需要设置`return_multi_modal_inputs: False`。

4. **异步模式**: 使用工具调用时必须使用异步rollout模式（`mode: async`）。
