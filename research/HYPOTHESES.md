# 假设

H-001 CURRENT_EVIDENCE_AGAINST：仅限原中部模块、alpha=1.5、固定模型及R5门槛；不推广至其他层。

H-002 UNTESTED：恢复累计输出可能部分缓解任务损失；EXP-003未满足门槛。

H-003 INSUFFICIENT_EVIDENCE：相近性能和活动预算下不同有效扇出可能产生不同SOPs。

H-004 CURRENT_EVIDENCE_SUPPORTS：当前模型前部conv1.2、alpha=1.5在独立样本中产生实际幅度累计输出偏差及任务损失增加。

H-005 INSUFFICIENT_EVIDENCE：若H-004获支持，固定目标模块动力学的参考幅值补偿可能缓解任务损失，并优于一般幅值调整。

H-006 INSUFFICIENT_EVIDENCE：在相近全网络SOPs预算下，动态重算真实前向边际代价的阈值分配方法，相对统一缩放与静态排序具有更好的任务性能权衡；仅为先导候选。

H-007 CURRENT_EVIDENCE_AGAINST：仅针对当前协议下预期普通微调SOP回升的预测；预算约束干预效果不可识别。原假设：在本协议中，固定阈值后的普通微调可能导致SOPs回升，显式预算约束可能缓解回升。

H-008 INSUFFICIENT_EVIDENCE：原假设：在相近实际SOPs预算下，自蒸馏可能改善高活动稀疏AT-LIF的任务性能。

H-009 UNTESTED：固定权重阈值与输入时，BN batch模式相对eval模式可能导致具有实际幅度的全网络SOP变化。
H-010 UNTESTED：在相近计算预算下，逐层推理状态BN总体统计校准可能改善任务性能。
