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


### ROUND-018 theory review
H-016 SUPPORTED only for nominal/effective fanout mismatch at current model and operating point; task and training impact untested. H-018 PARTIALLY_SUPPORTED: epoch60 direction passed but 5% magnitude missed; epoch300 activity term negative. H-017 UNTESTED. H-019 UNTESTED: fixed weights and BN running statistics, norm-matched MP-EFR may reduce target consumer SOP more than G0 and permuted G2 at fixed q=.02.


### ROUND-019 theory review
H-019 SUPPORTED only for fixed U2 epoch300, two target modules, rho=.25, q=.02, designated data and single eval local intervention. H-017 INSUFFICIENT_EVIDENCE; formal training impact UNTESTED. H-016 SUPPORTED and H-018 PARTIALLY_SUPPORTED retain prior scopes and negative evidence.


### ROUND-020 theory decision
H-017 INSUFFICIENT_EVIDENCE: EXP-021 matched G1 vs G0 but failed preregistered accuracy gain; missing fair comparator is no longer the explanation. H-016 SUPPORTED, H-018 PARTIALLY_SUPPORTED, H-019 SUPPORTED retain scopes. H-020 UNTESTED: an actual target-layer G1 activity gradient conflicts with task gradient, cos < −1e−6. H-021 UNTESTED: protected effective-fanout direction may improve matched-SOP task performance vs G0/G1/P0.


### ROUND-021 theory decision
H-020 SUPPORTED only current model/data/surrogate protocol; H-021 CONTRADICTED only preregistered fixed-model local intervention accuracy prediction. H-017 INSUFFICIENT_EVIDENCE. H-022 UNTESTED: layer exposure heterogeneity for fixed K in prespecified 3 layers; H-023 UNTESTED: fixed SEAWP training may improve preregistered efficiency or performance vs W0 and W2.


### ROUND-022 theory decision
H-022 SUPPORTED scoped U2 epoch100 exposure heterogeneity; H-023 CONTRADICTED only fixed alpha/model/101–110 and preregistered gates, retaining the EXP-023 pretraining unequal-tail admission deviation. H-024 UNTESTED: exact fixed-deletion output reconstruction under specified structure. H-025 UNTESTED: conditional marginal selection may improve matched 25% structured pruning performance.


## ROUND-023 theory review archive
H-024 SUPPORTED only for the investigated fixed model and linear readout. H-025 PARTIALLY_SUPPORTED: M5 improves over S3 but does not pass prespecified performance gates against S1/S2. EXP-024 overall progression remains FAILED. Q-013A/B are untested questions, not proven defects; H-017 remains UNTESTED.


## ROUND-024 preregistration
H-026 UNTESTED: 50% R1 accuracy drops ≥1 percentage point versus B0. H-027 UNTESTED: 50% R3 achieves all fixed D_cal reconstruction, validation accuracy/CE, equal-parameter, support and SOP gates. 75% is auxiliary. Scientific decisions require theory review.

## ROUND-025 theory decision
EXP-026 remains INVALID; evidence for H-026/H-027 INVALID_EVIDENCE, preserving prior UNTESTED history. EXP-027 is a new independent engineering replication, with the original 50% primary and 75% secondary gates unchanged; scientific decision remains with the theoretical researcher.

## ROUND-026 theory decision and preregistration
H-026 CURRENT_EVIDENCE_AGAINST scoped to the 50% condition and >=1pp loss prediction; H-027 CONTRADICTED scoped to fixed-support ridge performance, while D_cal logits MSE decrease subclaim supported. Preserve EXP-026 INVALID and EXP-027 VALID. H-028/H-029 UNTESTED: original fixed 5-epoch F1-versus-P0 and F1-versus-F0 preregistered gates; see ROUND-026-TASK.md.

## ROUND-027 theory decision and preregistration
H-028 PARTIALLY_SUPPORTED scoped: positive F1 recovery but >=1pp joint gate failed. H-029 INSUFFICIENT_EVIDENCE: F1 CE lower but no required accuracy advantage over F0. Both original progression gates failed; no established severe accuracy-sparsity conflict. H-030/H-031/H-032 UNTESTED, see ROUND-027-TASK.md; EXP-026 INVALID, EXP-027 VALID and EXP-028 repaired evaluation VALID remain preserved.
