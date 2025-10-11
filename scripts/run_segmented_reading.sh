#!/bin/bash

# Segmented Reading Training Script

set -e
set -x

# Set environment variables
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

# Run training
python3 -m verl.trainer.main_ppo \
  --config-path=verl/verl/trainer/config \
  --config-name=segmented_reading \
  hydra.run.dir=/user/luzhenyan \
  data.train_files=/user/luzhenyan/data/segmented_docs/train.parquet \
  data.val_files=/user/luzhenyan/data/segmented_docs/val.parquet \
  data.train_batch_size=16 \
  data.max_prompt_length=2048 \
  data.max_response_length=1024 \
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
  trainer.default_local_dir=/user/luzhenyan/checkpoints \
  2>&1 | tee /user/luzhenyan/segmented_reading.log
