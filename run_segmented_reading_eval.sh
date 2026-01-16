#!/bin/bash
# 基于GSM8K配置的分段阅读训练脚本
# 内存优化版本 - 解决CUDA内存不足问题

# set -x  # 已禁用详细调试输出

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_DEDUP_LOGS=0
export VERL_LOGGING_LEVEL=INFO
export RAY_TMPDIR=/var/luzhenyan/tmp
export TENSORBOARD_DIR=/var/luzhenyan/tensorboard/eval_checkpoint_600

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    hydra.run.dir=/var/luzhenyan/outputs \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=2 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/var/wangyicheng/models/Qwen2.5-1.5B-Instruct-segread-gs600 \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console','tensorboard']" \
    trainer.project_name='segmented_reading' \
    trainer.experiment_name='qwen2.5-1.5b_segmented_reading_eval_100' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_training_steps=0\
    data.train_files=/home/wangyicheng/data/hotpotqa_eval/eval_100/eval_verl.parquet \
    data.val_files=/home/wangyicheng/data/hotpotqa_eval/eval_100/eval_verl.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/verl/utils/tools/segmented_reading_tools.yaml" \
    trainer.total_epochs=1 $@


