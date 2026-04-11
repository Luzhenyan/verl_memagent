# 实验记录

## 进行中

| ID | 节点 | 脚本 | 启动时间 | Git Commit | Dataset | Window Size | Chunk Size | Summary Interval | Length Penalty | Others | 当前进度 | 当前 score | 备注 |
|----|------|------|----------|------------|---------|-------------|------------|-----------------|----------------|--------|----------|------------|------|
| exp-006 | thu_a100 | run_sw_docqa_grpo.sh | 2026-04-12 00:44 | 9026dda | DocQA_RL_1.6K (doc len 18k~24k tokens) | 12000 | 1600 | 4 (dataset default) | factor=0.0 (off) | max_resp=37500, rollout_n=4, batch=2 | step 101（生成中）| - | 日志: train_sw_docqa_2026-04-12_00-44-19.log；从 global_step_100 恢复 |
| exp-004 | thu_a100_node24 | run_sw.sh | 2026-04-10 15:45 | 9026dda | hotpotqa_notes_100k | 6000 | 1200 | 3 (env var) | factor=0.0 (off) | max_resp=37500, rollout_n=4, batch=2 | step 290/1000 | 0.125~0.375 | clip_ratio 0.375~0.625（较初期 1.0 已好转，仍偏高）|

## 已完成 / 已停止

| ID | 节点 | 脚本 | 时间段 | Git Commit | Dataset | Window Size | Chunk Size | Summary Interval | Length Penalty | Others | 最终进度 | 最终 score | 停止原因 |
|----|------|------|--------|------------|---------|-------------|------------|-----------------|----------------|--------|----------|------------|----------|
| exp-005 | thu_a100 | run_sw_docqa_grpo.sh | 2026-04-11 14:55 ~ 14:5x | 9026dda | DocQA_RL_1.6K (doc len 18k~24k tokens) | 6000 | 1600 | 4 (dataset default) | factor=0.0 (off) | max_resp=37500, rollout_n=4, batch=2 | step 100（未推进）| - | window size 漏改成 12000，手动停止，替换为 exp-006 |
| exp-003 | thu_a100 | run_sw_docqa_grpo.sh | 2026-04-10 17:34 ~ 2026-04-11 02:28 | 9026dda | DocQA_RL_1.6K (doc len 18k~24k tokens) | 6000 | 1600 | 4 (dataset default) | factor=0.0 (off) | max_resp=37500, rollout_n=4, batch=2 | step 126/500 | 0.625 (step 100) | SSH 断开导致 SIGTERM（未 nohup），已保存 global_step_100 |
| exp-002 | thu_a100_node2 | run_sw.sh | 2026-04-09 18:53 ~ 21:48 | - | hotpotqa_notes_100k | 8000 | 1200 | 4 (dataset default) | 未知 | 从 global_step_600 恢复 | step 712/1000 | 0.875 | 原因未知，进程消失 |
| exp-001 | thu_a100 | run_sw_docqa_grpo.sh | 2026-04-09 17:46 ~ 2026-04-10 | - | DocQA_RL_1.6K | 未知 | 1600 | 4 (dataset default) | 未知 | - | step 4+ | 0.5 | 被手动 kill（重启实验） |
