#!/bin/bash
# GPU机器部署脚本
# 用于在另一台有GPU的机器上运行VERL训练

set -e
set -x

echo "=== VERL GPU训练部署脚本 ==="

# 1. 检查GPU环境
echo "检查GPU环境..."
nvidia-smi || { echo "错误：未检测到GPU或nvidia-smi不可用"; exit 1; }

# 2. 检查Python环境
echo "检查Python环境..."
python3 --version
pip3 --version

# 3. 安装依赖
echo "安装依赖..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers datasets pandas pyarrow
pip3 install ray[default] hydra-core omegaconf
pip3 install vllm accelerate bitsandbytes

# 4. 设置环境变量
echo "设置环境变量..."
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_URL=https://hf-mirror.com
export PYTHONPATH=/path/to/verl:$PYTHONPATH  # 需要修改为实际路径

# 5. 创建数据目录
echo "创建数据目录..."
mkdir -p /path/to/data/triviaqa_docs  # 需要修改为实际路径

# 6. 下载或复制数据文件
echo "准备数据文件..."
# 如果有数据文件，复制到目标目录
# cp /path/to/source/train.parquet /path/to/data/triviaqa_docs/
# cp /path/to/source/val.parquet /path/to/data/triviaqa_docs/

# 或者重新生成数据
echo "重新生成TriviaQA数据..."
python3 scripts/prepare_triviaqa.py

# 7. 检查配置文件
echo "检查配置文件..."
ls -la verl/verl/trainer/config/segmented_reading.yaml

# 8. 运行训练
echo "开始训练..."
python3 verl/trainer/main_ppo.py --config-name=segmented_reading

echo "部署完成！"






