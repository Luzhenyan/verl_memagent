# VERL GPU训练部署指南

## 概述
本指南帮助你在另一台有GPU的机器上部署和运行VERL训练。

## 前置要求

### 硬件要求
- NVIDIA GPU（推荐RTX 3090/4090或A100/H100）
- 至少16GB GPU内存
- 至少32GB系统内存
- 至少100GB可用磁盘空间

### 软件要求
- Ubuntu 20.04+ 或 CentOS 7+
- Python 3.8+
- CUDA 11.8+
- NVIDIA驱动 450+

## 部署步骤

### 1. 环境检查
```bash
# 检查GPU
nvidia-smi

# 检查Python
python3 --version
pip3 --version

# 检查CUDA
nvcc --version
```

### 2. 安装依赖
```bash
# 安装PyTorch（根据你的CUDA版本选择）
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip3 install transformers datasets pandas pyarrow
pip3 install ray[default] hydra-core omegaconf
pip3 install vllm accelerate bitsandbytes
```

### 3. 代码部署
```bash
# 方法1：克隆代码
git clone https://github.com/your-repo/verl.git
cd verl

# 方法2：复制代码（如果有本地代码）
scp -r /path/to/local/verl user@gpu-machine:/path/to/destination/
```

### 4. 数据准备
```bash
# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_URL=https://hf-mirror.com
export PYTHONPATH=/path/to/verl:$PYTHONPATH

# 创建数据目录
mkdir -p /path/to/data/triviaqa_docs

# 方法1：重新生成数据
python3 scripts/prepare_triviaqa.py

# 方法2：复制已有数据
scp /path/to/source/train.parquet user@gpu-machine:/path/to/data/triviaqa_docs/
scp /path/to/source/val.parquet user@gpu-machine:/path/to/data/triviaqa_docs/
```

### 5. 配置调整
根据GPU配置调整`verl/verl/trainer/config/segmented_reading.yaml`：

```yaml
# 根据GPU内存调整
actor_rollout_ref:
  rollout:
    gpu_memory_utilization: 0.8  # 根据GPU内存调整
    tensor_model_parallel_size: 1  # 单GPU设为1

# 根据GPU数量调整
trainer:
  n_gpus_per_node: 1  # 根据实际GPU数量
  nnodes: 1

# 根据GPU内存调整batch size
data:
  train_batch_size: 8  # 根据GPU内存调整
actor_rollout_ref:
  actor:
    ppo_micro_batch_size_per_gpu: 1  # 根据GPU内存调整
```

### 6. 运行训练
```bash
# 设置PYTHONPATH
export PYTHONPATH=/path/to/verl:$PYTHONPATH

# 运行训练
python3 verl/trainer/main_ppo.py --config-name=segmented_reading
```

## 性能优化建议

### GPU内存优化
- 如果GPU内存不足，减少`train_batch_size`和`ppo_micro_batch_size_per_gpu`
- 调整`gpu_memory_utilization`（0.6-0.9之间）
- 使用更小的模型（如0.5B而不是7B）

### 训练速度优化
- 使用更快的存储（SSD/NVMe）
- 增加`train_batch_size`（在内存允许的情况下）
- 使用多GPU训练（如果有多个GPU）

### 网络优化
- 使用HF镜像加速模型下载
- 预下载模型到本地缓存

## 监控和调试

### 监控GPU使用
```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 监控训练日志
tail -f /path/to/outputs/verl_demo.log
```

### 常见问题
1. **CUDA out of memory**：减少batch size或使用更小的模型
2. **ModuleNotFoundError**：检查PYTHONPATH设置
3. **Ray初始化失败**：检查端口占用和防火墙设置

## 数据文件说明

### 生成的数据文件
- `train.parquet`：训练数据（约600MB，10000个样本）
- `val.parquet`：验证数据（约32MB，1000个样本）

### 数据格式
```python
{
    'prompt': List[Dict],      # 用户消息
    'response': List[Dict],    # 助手消息
    'question': str,           # 问题
    'answer': str,             # 答案
    'context': str,            # 文档内容（长文本）
    'extra_info': Dict         # 额外信息
}
```

## 预期结果

### 训练时间
- 单GPU RTX 3090：约2-4小时
- 单GPU A100：约1-2小时

### 输出文件
- 模型检查点：`/path/to/checkpoints/`
- 训练日志：`/path/to/outputs/`
- 配置文件：`/path/to/configs/`

## 联系支持
如果遇到问题，请检查：
1. GPU驱动和CUDA版本
2. Python环境和依赖版本
3. 数据文件完整性
4. 配置文件路径设置






