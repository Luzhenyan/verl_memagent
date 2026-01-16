#!/bin/bash
# 基础TriviaQA RL训练脚本
# 简化版本 - 不使用多轮对话、分段或工具调用

# set -x  # 已禁用详细调试输出

ulimit -n 65535

PROJECT_DIR="$(pwd)"
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_DEDUP_LOGS=0
export VERL_LOGGING_LEVEL=INFO
export RAY_TMPDIR=/var/luzhenyan/tmp
export TENSORBOARD_DIR=/var/luzhenyan/tensorboard/hotpotqa_boxed_$(date +%Y%m%d_%H%M%S)

python3 -m verl.trainer.main_ppo \
    hydra.run.dir=/var/luzhenyan/outputs \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=2 \
    data.max_prompt_length=31744 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/var/wangyicheng/models/Qwen2.5-7B-Instruct/snapshots/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name='hotpotqa_boxed' \
    trainer.experiment_name='qwen2.5-7b_hotpotqa_boxed_basic' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=500 \
    data.train_files=/home/wangyicheng/data/hotpotqa_boxed_1000/train.parquet \
    data.val_files=/home/wangyicheng/data/hotpotqa_boxed_1000/train.parquet \
    trainer.total_epochs=3 \
    2>&1 | tee /home/wangyicheng/verl_memagent/logs/train_$(date +%F_%H-%M-%S).log $@


