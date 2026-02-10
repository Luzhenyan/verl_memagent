#!/bin/bash
# SW（手动分段）+ GRPO：复刻 eval 分段阅读/总结流程（无工具、固定模板、多轮）
#
# 参考：run_simple_7B.sh
# 依赖数据：/var/wangyicheng/data/hotpotqa_train_32k_sw.parquet（已包含 agent_name=streaming_chunk_agent）
# 依赖模型：/var/wangyicheng/models/Qwen3-8B
#
# 用法：
#   bash run_sw.sh
#   CUDA_VISIBLE_DEVICES=7 bash run_sw.sh
#   TRAIN_FILE=... VAL_FILE=... bash run_sw.sh

set -euo pipefail

ulimit -n 65535

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

# Hydra config 里有相对 searchpath（如 file://verl/trainer/config），必须确保 cwd 在仓库根目录
cd "$PROJECT_DIR"

# python 优先使用 verl_Mem 环境
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "/home/wangyicheng/.conda/envs/verl_Mem/bin/python" ]]; then
  PYTHON_BIN="/home/wangyicheng/.conda/envs/verl_Mem/bin/python"
fi

# quick dependency check
$PYTHON_BIN -c "import ray" >/dev/null 2>&1 || {
  echo "[run_sw] ERROR: python cannot import ray. Please install: pip install 'ray[default]'" >&2
  exit 1
}

export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"

# 允许 vLLM 窗口超过模型原生的 40k 限制 (V0 后端通常能通过 RoPE 外推跑通)
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# 解决磁盘满额问题：将所有临时目录重定向到 /var/luzhenyan/tmp
export RAY_TMPDIR="/var/luzhenyan/tmp"
export TMPDIR="/var/luzhenyan/tmp"
export TEMP="/var/luzhenyan/tmp"
export TMP="/var/luzhenyan/tmp"
mkdir -p /var/luzhenyan/tmp

export TENSORBOARD_DIR="${TENSORBOARD_DIR:-/var/luzhenyan/tensorboard/hotpotqa_sw_$(date +%Y%m%d_%H%M%S)}"

# Debug logs (可用环境变量覆盖)
# - VERL_SW_DEBUG=1: 打印 streaming_chunk_agent_loop 里程碑日志
# - VERL_SW_DEBUG_VERBOSE=1: 打印每次 generate 的耗时/token 数（很啰嗦）
# - VERL_VLLM_ASYNC_DEBUG=1: 打印 vllm_async_server init_engine 里程碑耗时
export VERL_SW_DEBUG="${VERL_SW_DEBUG:-1}"
export VERL_SW_DEBUG_VERBOSE="${VERL_SW_DEBUG_VERBOSE:-0}"
export VERL_SW_LOG_EVERY="${VERL_SW_LOG_EVERY:-5}"
export VERL_VLLM_ASYNC_DEBUG="${VERL_VLLM_ASYNC_DEBUG:-1}"
export VERL_SW_MAX_NEW_PER_CALL="${VERL_SW_MAX_NEW_PER_CALL:-2048}"

# 数据/模型
TRAIN_FILE="${TRAIN_FILE:-/var/wangyicheng/data/hotpotqa_train_32k_sw.parquet}"
VAL_FILE="${VAL_FILE:-/var/wangyicheng/data/hotpotqa_train_32k_sw.parquet}"
MODEL_PATH="${MODEL_PATH:-/var/wangyicheng/models/Qwen3-8B}"

# 自动检测 GPU 数量并设置 N_GPUS_PER_NODE
# if [[ -z "${N_GPUS_PER_NODE:-}" ]]; then
#   # 统计 CUDA_VISIBLE_DEVICES 中的逗号数量并加 1
#   N_GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr -cd ',' | wc -c)
#   N_GPUS_PER_NODE=$((N_GPUS_PER_NODE + 1))
# fi
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
echo "[run_sw] Using N_GPUS_PER_NODE=$N_GPUS_PER_NODE"

# 训练参数（先给一个可跑的默认；可通过环境变量覆盖）
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2500}"     # prompt 只有 question；长文在 context 字段里由 agent loop 手动切块
MAX_RESP_LEN="${MAX_RESP_LEN:-90000}"        # 恢复较大的响应长度
MAX_MODEL_LEN="${MAX_MODEL_LEN:-100000}"       # 恢复较大的上下文长度
TOTAL_STEPS="${TOTAL_STEPS:-1000}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
ROLLOUT_N="${ROLLOUT_N:-4}"

# PPO / update 相关：当使用更多 GPU（例如 8 卡）时，ppo_mini_batch_size 太小可能会在 worker 初始化时被归一化成 0。
# 经验上：至少设为 >= N_GPUS_PER_NODE（或更大）更稳。
PPO_MINI_BSZ="${PPO_MINI_BSZ:-2}"

# 自动调整 TRAIN_BATCH_SIZE 以适配 GPU 数量
# 逻辑：(TRAIN_BATCH_SIZE * ROLLOUT_N) 必须能被 N_GPUS_PER_NODE 整除
while [[ $(( (TRAIN_BATCH_SIZE * ROLLOUT_N) % N_GPUS_PER_NODE )) != 0 ]]; do
  TRAIN_BATCH_SIZE=$((TRAIN_BATCH_SIZE + 1))
done
echo "[run_sw] Adjusted TRAIN_BATCH_SIZE to $TRAIN_BATCH_SIZE for compatibility with $N_GPUS_PER_NODE GPUs (Total samples: $((TRAIN_BATCH_SIZE * ROLLOUT_N)))"

TP="${TP:-8}"


LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
mkdir -p "$LOG_DIR"

set -x

PROJECT_NAME="${PROJECT_NAME:-hotpotqa_100k_sw}"
EXP_NAME="${EXP_NAME:-qwen3-8b}"
# 根分区 / 当前已满（df 显示 100%），checkpoint 写到工程目录下很容易失败。
# 默认把 checkpoint 放到 /var（通常更大）；如需自定义可通过环境变量覆盖 CKPT_DIR。
CKPT_DIR="${CKPT_DIR:-/var/luzhenyan/checkpoints}"

$PYTHON_BIN -m verl.trainer.main_ppo \
  --config-path="$CONFIG_PATH" \
  --config-name='gsm8k_multiturn_grpo' \
  hydra.run.dir=/var/luzhenyan/outputs \
  algorithm.adv_estimator=grpo \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LEN" \
  data.max_response_length="$MAX_RESP_LEN" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BSZ" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.tool_config_path=null \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger="['console','tensorboard']" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXP_NAME" \
  trainer.default_local_dir="${CKPT_DIR}/${PROJECT_NAME}/${EXP_NAME}" \
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
  trainer.nnodes=1 \
  trainer.device=cuda \
  trainer.val_before_train=False \
  trainer.save_freq=50 \
  trainer.test_freq=-1 \
  trainer.total_training_steps="$TOTAL_STEPS" \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.val_batch_size=1 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  2>&1 | tee "$LOG_DIR/train_sw_$(date +%F_%H-%M-%S).log" $@


