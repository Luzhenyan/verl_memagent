##数据集
HotpotQA

##任务
多跳推理

##模型能力
分段阅读context （预先划分）
更新总结
给出回答 （指定步数）

##奖励函数
HotpotQA数据集中提供了supporting_facts，判断中间总结中是否包含supporting_facts
最终答案


单条数据可能太短了，可以扩充一下，用wikipage
使用最简单的单跳QA




0903 update
##数据集
Triviaqa.wikipedia
context长度 3w-12w
进行数据集分段，2048个字符为一段

##任务
单跳推理

##当前进展
配置文件较为复杂
还没有跑通



