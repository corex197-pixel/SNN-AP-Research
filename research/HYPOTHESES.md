# 假设

H-001 CURRENT_EVIDENCE_AGAINST：仅限原中部模块、alpha=1.5、固定模型及R5门槛；不推广至其他层。

H-002 UNTESTED：恢复累计输出可能部分缓解任务损失；EXP-003未满足门槛。

H-003 INSUFFICIENT_EVIDENCE：相近性能和活动预算下不同有效扇出可能产生不同SOPs。

H-004 CURRENT_EVIDENCE_SUPPORTS：当前模型前部conv1.2、alpha=1.5在独立样本中产生实际幅度累计输出偏差及任务损失增加。

H-005 INSUFFICIENT_EVIDENCE：若H-004获支持，固定目标模块动力学的参考幅值补偿可能缓解任务损失，并优于一般幅值调整。

H-006 INSUFFICIENT_EVIDENCE：在相近全网络SOPs预算下，动态重算真实前向边际代价的阈值分配方法，相对统一缩放与静态排序具有更好的任务性能权衡；仅为先导候选。

H-007 CURRENT_EVIDENCE_AGAINST：仅针对当前协议下预期普通微调SOP回升的预测；预算约束干预效果不可识别。原假设：在本协议中，固定阈值后的普通微调可能导致SOPs回升，显式预算约束可能缓解回升。

H-008 INSUFFICIENT_EVIDENCE：原假设：在相近实际SOPs预算下，自蒸馏可能改善高活动稀疏AT-LIF的任务性能。

H-009 CURRENT_EVIDENCE_SUPPORTS：仅限当前模型、固定500校准样本、batch128及模式干预。固定权重阈值与输入时，BN batch模式相对eval模式可能导致具有实际幅度的全网络SOP变化。
H-010 CURRENT_EVIDENCE_AGAINST：仅限主要20%预算相对C0改善预测，相对C3仍证据不足。在相近计算预算下，逐层推理状态BN总体统计校准可能改善任务性能。

H-011 INSUFFICIENT_EVIDENCE：原筛选失败且梯度规则未产生可分辨的最终模型差异。原假设：在相同初始化、训练和稀疏化规则及近似SOP预算下，M4优于G0、G1及简单阈值缩放。

H-012 UNTESTED：由mean恢复sum等价聚合可能产生可分辨的共享阈值优化器更新。
H-013 UNTESTED：有效阈值任务更新下，ARTG可能比其他代理因子获得更好的精度—SOPs权衡。

### ROUND-013 preregistration
H-012: SUPPORTED (maps the researcher-provided CURRENT_EVIDENCE_SUPPORTS to the AGENTS.md enum; restricted to the tested model, checkpoint, data and SGD protocol). H-013: INSUFFICIENT_EVIDENCE; retain all exploratory 30% observations. H-014: UNTESTED, feasible low-loss genuine internal structural redundancy with the specified parameter/SOP and accuracy gates. H-015: UNTESTED, direct loss-based channel choice versus FR and L1 with prespecified accuracy and SOP gates. Formal status awaits theory review.

### ROUND-014 theory review
H-014: CONTRADICTED only for EXP-014's three specified modules, 25% internal channel removal without fine-tuning and preregistered pilot gate; no universal claim about SNN structural redundancy. H-015: CONTRADICTED only for the fixed samples, single-channel loss ranking and simultaneous removal protocol. H-012 remains SUPPORTED within its original scoped evidence; H-013 remains INSUFFICIENT_EVIDENCE. All older hypothesis history and boundaries remain unchanged. Q-010 awaits deeper paper analysis; no new validated performance hypothesis.

### ROUND-015 preregistration
H-016 UNTESTED: fixed nominal activity fanout may differ from actual local SOP cost when effective connections change; synthetic validation does not establish importance in trained networks. H-017 UNTESTED: effective-connectivity-aware penalties may improve matched-SOP task performance; no training authorized this round.

ROUND-015 observations: H-016 synthetic fixed-mask/fixed-activity mechanism checks passed, limited to the specified conditions; scientific status stays UNTESTED pending theoretical review. The UniSparse paper itself explicitly specifies initial-structure approximation. H-017 remains UNTESTED and no ECAS training was run.

### ROUND-016 theory decision
H-016 PARTIALLY_SUPPORTED: fixed-activity/fixed-mask mathematical and synthetic mechanism supported; actual trained-network magnitude and task effects remain untested. H-017 UNTESTED; R64 is theory only, no ECAS training. Previous hypotheses and their scopes are retained.

### ROUND-017 preregistration
H-016 PARTIALLY_SUPPORTED, restricted to synthetic mechanisms and descriptive EXP-017 epoch20 trained-model fanout; optimization and task effects unresolved. H-017 UNTESTED. H-018 UNTESTED: in U2 vs U0, ΔC_W<0 and ΔC_S>0 under R69–R73; primary early checkpoint epoch60, exploratory practical magnitude ΔC_S/C00>=0.05, without causal claims.
