# 假设

H-001 CURRENT_EVIDENCE_AGAINST：仅限原中部模块、alpha=1.5、固定模型及R5门槛；不推广至其他层。

H-002 UNTESTED：恢复累计输出可能部分缓解任务损失；EXP-003未满足门槛。

H-003 INSUFFICIENT_EVIDENCE：相近性能和活动预算下不同有效扇出可能产生不同SOPs。

H-004 CURRENT_EVIDENCE_SUPPORTS：当前模型前部conv1.2、alpha=1.5在独立样本中产生实际幅度累计输出偏差及任务损失增加。

H-005 INSUFFICIENT_EVIDENCE：若H-004获支持，固定目标模块动力学的参考幅值补偿可能缓解任务损失，并优于一般幅值调整。

H-006 UNTESTED：在相近全网络SOPs预算下，动态重算真实前向边际代价的阈值分配方法，相对统一缩放与静态排序具有更好的任务性能权衡；仅为先导候选。
