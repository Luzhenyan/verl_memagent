#!/bin/bash
# run_sw_docqa_dapo.sh - DocQA 混合数据集（doc-qa/doc-mc/doc-math）的 SW+DAPO 训练
#
# 依赖：先运行 prepare_docqa_sw.py 生成训练数据：
#   python prepare_docqa_sw.py \
#       --input  /var/luzhenyan/data/DocQA_RL_1.6K_train.parquet \
#       --output /var/luzhenyan/data/docqa_train_sw.parquet
#
# 用法：
#   bash run_sw_docqa_dapo.sh
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash run_sw_docqa_dapo.sh
#   TRAIN_FILE=... VAL_FILE=... bash run_sw_docqa_dapo.sh

set -euo pipefail

ulimit -n 65535

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "/home/wangyicheng/.conda/envs/verl_Mem/bin/python" ]]; then
  PYTHON_BIN="/home/wangyicheng/.conda/envs/verl_Mem/bin/python"
fi

$PYTHON_BIN -c "import ray" >/dev/null 2>&1 || {
  echo "[run_sw_docqa_dapo] ERROR: python cannot import ray. Please install: pip install 'ray[default]'" >&2
  exit 1
}

export PYTHONPATH="$PROJECT_DIR/examples${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VERL_USE_LOCALHOST_MASTER="${VERL_USE_LOCALHOST_MASTER:-1}"

export RAY_TMPDIR="/var/luzhenyan/tmp"
export TMPDIR="/var/luzhenyan/tmp"
export TEMP="/var/luzhenyan/tmp"
export TMP="/var/luzhenyan/tmp"
mkdir -p /var/luzhenyan/tmp

# -------------------------------------------------------
# Hydra 输出目录（含 main_dapo.log、.hydra 配置快照等）
# /var/luzhenyan/outputs 在某些用户/容器下可能不可写，因此自动回退到项目内目录。
# -------------------------------------------------------
DEFAULT_HYDRA_RUN_DIR="/var/luzhenyan/outputs"
if mkdir -p "$DEFAULT_HYDRA_RUN_DIR" 2>/dev/null && [[ -w "$DEFAULT_HYDRA_RUN_DIR" ]]; then
  HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$DEFAULT_HYDRA_RUN_DIR}"
else
  HYDRA_RUN_DIR="${HYDRA_RUN_DIR:-$PROJECT_DIR/outputs}"
  mkdir -p "$HYDRA_RUN_DIR"
  echo "[run_sw_docqa_dapo] NOTE: '$DEFAULT_HYDRA_RUN_DIR' is not writable; using HYDRA_RUN_DIR=$HYDRA_RUN_DIR"
fi

export TENSORBOARD_DIR="${TENSORBOARD_DIR:-/var/luzhenyan/tensorboard/docqa_sw_dapo_$(date +%Y%m%d_%H%M%S)}"

export VERL_SW_DEBUG="${VERL_SW_DEBUG:-1}"
export VERL_SW_DEBUG_VERBOSE="${VERL_SW_DEBUG_VERBOSE:-0}"
export VERL_SW_LOG_EVERY="${VERL_SW_LOG_EVERY:-5}"
export VERL_VLLM_ASYNC_DEBUG="${VERL_VLLM_ASYNC_DEBUG:-1}"
export VERL_SW_MAX_NEW_PER_CALL="${VERL_SW_MAX_NEW_PER_CALL:-2048}"
export VERL_SW_AVG_TURN_PENALTY_START="${VERL_SW_AVG_TURN_PENALTY_START:-500}"
export VERL_SW_AVG_TURN_PENALTY_MAX="${VERL_SW_AVG_TURN_PENALTY_MAX:-1000}"
export VERL_SW_AVG_TURN_PENALTY_FACTOR="${VERL_SW_AVG_TURN_PENALTY_FACTOR:-1.0}"

# -------------------------------------------------------
# 若 docqa_train_sw.parquet 不存在，自动运行预处理脚本
# -------------------------------------------------------
DOCQA_TRAIN_RAW="${DOCQA_TRAIN_RAW:-/var/luzhenyan/data/DocQA_RL_1.6K_train.parquet}"
DOCQA_TRAIN_SW="${DOCQA_TRAIN_SW:-/var/luzhenyan/data/docqa_train_sw.parquet}"
DOCQA_VAL_RAW="${DOCQA_VAL_RAW:-/var/luzhenyan/data/DocQA_RL_1.6K_test.parquet}"
DOCQA_VAL_SW="${DOCQA_VAL_SW:-/var/luzhenyan/data/docqa_val_sw.parquet}"

if [[ ! -f "$DOCQA_TRAIN_SW" ]]; then
  echo "[run_sw_docqa_dapo] 训练集 SW 格式不存在，开始预处理..."
  VAL_ARGS=""
  if [[ -f "$DOCQA_VAL_RAW" ]]; then
    VAL_ARGS="--val_input $DOCQA_VAL_RAW --val_output $DOCQA_VAL_SW"
  fi
  $PYTHON_BIN "$PROJECT_DIR/prepare_docqa_sw.py" \
      --input "$DOCQA_TRAIN_RAW" \
      --output "$DOCQA_TRAIN_SW" \
      $VAL_ARGS
  echo "[run_sw_docqa_dapo] 预处理完成：$DOCQA_TRAIN_SW"
fi

# 选择验证集：优先用预处理后的 docqa_val_sw.parquet，其次回退到训练集
if [[ -f "$DOCQA_VAL_SW" ]]; then
  DEFAULT_VAL_FILE="$DOCQA_VAL_SW"
else
  DEFAULT_VAL_FILE="$DOCQA_TRAIN_SW"
fi

TRAIN_FILE="${TRAIN_FILE:-$DOCQA_TRAIN_SW}"
VAL_FILE="${VAL_FILE:-$DEFAULT_VAL_FILE}"
MODEL_PATH="${MODEL_PATH:-/var/wangyicheng/models/Qwen3-8B}"

export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
echo "[run_sw_docqa_dapo] Using N_GPUS_PER_NODE=$N_GPUS_PER_NODE"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-2500}"
MAX_RESP_LEN="${MAX_RESP_LEN:-57500}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-60000}"
TOTAL_STEPS="${TOTAL_STEPS:-1000}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
ROLLOUT_N="${ROLLOUT_N:-4}"
PPO_MINI_BSZ="${PPO_MINI_BSZ:-2}"

while [[ $(( (TRAIN_BATCH_SIZE * ROLLOUT_N) % N_GPUS_PER_NODE )) != 0 ]]; do
  TRAIN_BATCH_SIZE=$((TRAIN_BATCH_SIZE + 1))
done
echo "[run_sw_docqa_dapo] Adjusted TRAIN_BATCH_SIZE to $TRAIN_BATCH_SIZE (Total samples: $((TRAIN_BATCH_SIZE * ROLLOUT_N)))"

GEN_BATCH_MULTIPLIER="${GEN_BATCH_MULTIPLIER:-3}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-$((TRAIN_BATCH_SIZE * GEN_BATCH_MULTIPLIER))}"

TP="${TP:-8}"
SP_SIZE="${SP_SIZE:-4}"
PPO_MAX_TOKEN_LEN="${PPO_MAX_TOKEN_LEN:-24576}"

CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
ENABLE_FILTER_GROUPS="${ENABLE_FILTER_GROUPS:-True}"
# NOTE: When using RM scores (rm_scores) with DAPO, per-sample `acc` is typically not populated.
# `seq_final_reward` is always available (derived from token_level_rewards) and works well for group filtering.
FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-seq_final_reward}"
MAX_NUM_GEN_BATCHES="${MAX_NUM_GEN_BATCHES:-8}"
ENABLE_OVERLONG_BUFFER="${ENABLE_OVERLONG_BUFFER:-True}"
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-10000}"
OVERLONG_PENALTY_FACTOR="${OVERLONG_PENALTY_FACTOR:-1.0}"

LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
mkdir -p "$LOG_DIR"

PROJECT_NAME="${PROJECT_NAME:-docqa_sw_dapo}"
EXP_NAME="${EXP_NAME:-qwen3-8b}"
CKPT_DIR="${CKPT_DIR:-/var/luzhenyan/checkpoints}"

set -x

$PYTHON_BIN -m recipe.dapo.main_dapo \
  hydra.run.dir="$HYDRA_RUN_DIR" \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.gen_batch_size="$GEN_BATCH_SIZE" \
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
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="$SP_SIZE" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.clip_ratio_low="$CLIP_RATIO_LOW" \
  actor_rollout_ref.actor.clip_ratio_high="$CLIP_RATIO_HIGH" \
  actor_rollout_ref.actor.loss_agg_mode="$LOSS_AGG_MODE" \
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
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.filter_groups.enable="$ENABLE_FILTER_GROUPS" \
  algorithm.filter_groups.metric="$FILTER_GROUPS_METRIC" \
  algorithm.filter_groups.max_num_gen_batches="$MAX_NUM_GEN_BATCHES" \
  reward_model.reward_manager=dapo \
  reward_model.overlong_buffer.enable="$ENABLE_OVERLONG_BUFFER" \
  reward_model.overlong_buffer.len="$OVERLONG_BUFFER_LEN" \
  reward_model.overlong_buffer.penalty_factor="$OVERLONG_PENALTY_FACTOR" \
  trainer.critic_warmup=0 \
  trainer.logger='["console","tensorboard"]' \
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
  data.val_batch_size=1 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  "$@" \
  2>&1 | tee "$LOG_DIR/train_sw_docqa_dapo_$(date +%F_%H-%M-%S).log"
