# 实验记录

## 进行中

| ID | 节点 | 脚本 | 启动时间 | Git Commit | 关键配置 | 当前进度 | 当前 score | 备注 |
|----|------|------|----------|------------|----------|----------|------------|------|
| exp-003 | thu_a100 | run_sw_docqa_grpo.sh | 2026-04-10 13:43 | 9026dda | docqa, sw=12000→6000, no penalty | step 4+ | - | 重启后进行中 |
| exp-004 | thu_a100_node24 | run_sw.sh | 2026-04-10 15:45 | 9026dda | hotpotqa, sw=6000, chunk_tokens=default, summary_interval=3, no penalty | 启动中 | - | PYTHON_BIN=/root/miniconda3/envs/verl_Mem/bin/python |

## 已完成 / 已停止

| ID | 节点 | 脚本 | 时间段 | Git Commit | 关键配置 | 最终进度 | 最终 score | 停止原因 |
|----|------|------|--------|------------|----------|----------|------------|----------|
| exp-002 | thu_a100_node2 | run_sw.sh | 2026-04-09 18:53 ~ 21:48 | - | hotpotqa, memory ability, 从 global_step_600 恢复 | step 712/1000 | 0.875 | 原因未知，进程消失 |
| exp-001 | thu_a100 | run_sw_docqa_grpo.sh | 2026-04-09 17:46 ~ 2026-04-10 | - | docqa | step 4+ | 0.5 | 被手动 kill（重启实验） |
