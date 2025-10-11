# 分段阅读RL训练使用指南

## 概述

这个项目基于VERL框架实现了分段阅读的强化学习训练，让模型学会：
- 分段读取长文档
- 为每段生成总结
- 基于已有总结决定下一步读什么
- 最终生成准确答案

## 快速开始

### 1. 准备数据

```bash
# 创建数据目录
mkdir -p /user/luzhenyan/data/segmented_docs

# 生成示例数据
cd verl
python scripts/prepare_segmented_data.py
```

### 2. 测试工具

```bash
# 测试阅读工具功能
python scripts/test_reading_tools.py
```

### 3. 开始训练

```bash
# 运行训练
bash scripts/run_segmented_reading.sh
```

## 工具说明

### ReadDocumentTool
- **功能**：读取文档的指定段落
- **参数**：`file_path`（文件路径）、`segment_index`（段落索引）
- **奖励**：0分（基础动作）

### WriteSummaryTool
- **功能**：为段落生成总结
- **参数**：`segment_content`（段落内容）、`summary`（总结）
- **奖励**：基于总结质量，质量>0.5时给3分

### UpdateCurrentSummaryTool
- **功能**：基于所有段落总结更新当前总结
- **参数**：`segment_summaries`（段落总结）、`question`（问题）、`current_summary`（当前总结）
- **奖励**：基于帮助程度，最高5分

### GenerateFinalAnswerTool
- **功能**：基于当前总结生成最终答案
- **参数**：`current_summary`（当前总结）、`question`（问题）、`final_answer`（最终答案）
- **奖励**：基于答案准确性，最高10分

## 训练流程

```
问题 → 读第1段 → 写总结1 → 更新当前总结 → 读第2段 → 写总结2 → 更新当前总结 → ... → 生成最终答案
```

## 配置说明

### 训练配置 (`verl/verl/trainer/config/segmented_reading.yaml`)

```yaml
trainer:
  project_name: "segmented_reading"
  experiment_name: "smart_reader"
  total_epochs: 50
  
data:
  train_batch_size: 16
  max_prompt_length: 2048
  max_response_length: 1024

actor_rollout_ref:
  model:
    path: "Qwen/Qwen2.5-0.5B-Instruct"
```

### 工具配置

```yaml
tools:
  reading_tools:
    enabled: true
    max_segment_length: 500
    max_segments_per_doc: 20
    min_segments_to_read: 2
    max_segments_to_read: 15
```

## 数据格式

训练数据为parquet格式，包含以下字段：

```json
{
  "file_path": "文档文件路径",
  "question": "待回答问题",
  "segments": ["段落1", "段落2", "段落3"],
  "segment_summaries": ["总结1", "总结2", "总结3"],
  "current_summary": "当前总结",
  "expected_answer": "期望答案",
  "relevant_segments": [0, 1, 2],
  "difficulty": "easy/medium/hard"
}
```

## 奖励机制

- **段落总结质量**：基于关键词覆盖率评估
- **当前总结帮助性**：基于与问题的相关性评估
- **最终答案准确性**：基于与问题的相关性和长度评估

## 监控训练

训练日志保存在 `/user/luzhenyan/segmented_reading.log`

```bash
# 查看训练进度
tail -f /user/luzhenyan/segmented_reading.log
```

## 模型输出

训练完成后，模型checkpoints保存在 `/user/luzhenyan/checkpoints/`

## 扩展功能

### 1. 添加更多工具
可以在 `verl/verl/tools/reading_tools.py` 中添加新的工具类。

### 2. 改进奖励函数
可以修改工具中的评估方法来改进奖励计算。

### 3. 增加数据源
可以修改 `scripts/prepare_segmented_data.py` 来支持更多数据格式。

## 故障排除

### 1. 文件路径问题
确保所有文件路径都指向正确的位置，特别是数据文件路径。

### 2. 内存不足
可以调整 `train_batch_size` 和 `gpu_memory_utilization` 参数。

### 3. 工具执行错误
检查工具的参数格式是否正确，确保JSON格式正确。

## 下一步

1. 增加更多文档类型支持
2. 改进分段算法
3. 优化奖励函数
4. 添加更多评估指标
