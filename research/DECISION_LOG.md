# 决策记录

DEC-001 — CONTINUE：主线 Q-001；Q-002 为竞争解释；Q-004 为独立备选；本轮不实现创新算法。动力学代数已提出，但真实网络任务作用尚无证据。

DEC-002 — REPAIR：ROUND-001 交付阶段性通过，EXP-001 数值通过但独立无捕获对照未完成。Q-001、Q-002、理论 v0.1.0 保持不变；已确认阈值梯度平方与 Batch 求和，不修改 baseline；EXP-004 准备 baseline 复现。

DEC-003 — REPAIR → CONTINUE：EXP-001 正式验收通过；Q-001 主线不变、H-001/H-002 UNTESTED。采用独立验证集与诊断池，修复测试集选模型和评估状态污染；保持 AT-LIF 核心算法；本轮允许经授权单卡资源测量，完整训练另需批准。

DEC-004 — CONTINUE：ROUND-003 基本验收，R13 有 CPU 工程证据；保留 AMP 跳步后原始手动阈值更新，完成 GPU 前置检查及训练日志后执行获批 EXP-004，Q-001 和 v0.1.1 不变。前置检查状态以实际记录为准。

ROUND-004 执行审计记录：正式训练部分完成（37/200），停止原因未知；不将其视作完整 baseline，不自动续训或运行 EXP-002/003。H-001/H-002 保持 UNTESTED；后续科研决策待理论研究员审查。

DEC-005 — REPAIR → CONTINUE：37轮已具备任务学习能力；两份检查点CPU可读且哈希吻合，无证据证明 AT-LIF 导致中断；从最新检查点恢复以保留投入，H-001/H-002 未验证。

DEC-006 — CONTINUE：EXP-004 Controlled Baseline 通过理论验收；H-001/H-002 尚未验证。按原方案开展 EXP-002，只有满足门槛方可执行 EXP-003；不新增算法，R14 是预注册辅助可恢复性上界。

DEC-007 — REFINE + REPLICATE：原H-001的当前反对证据保留，原主线不得作为普遍补偿不足继续推进；Q-001A仅针对前部探索发现开展固定独立样本验证，条件性运行反事实，不无限更换干预条件。

DEC-008 — PIVOT + IMPLEMENT：保留H-001/H-004限定条件证据与H-005证据不足，暂停幅值补偿主线；转向校准任务损失与实际SOPs联合驱动的TCAP候选活动剪枝。TCAP是工作名称，不宣称原创性；不训练。

## DEC-009 — PIVOT + IMPLEMENT
理论研究员确认H-006 INSUFFICIENT_EVIDENCE，停止当前TCAP动态贪心方案，不推广为该类方法普遍无效。转向AT-LIF固定阈值短期预算约束微调和自蒸馏，按ROUND-009预注册先导验证；H-001至H-005历史状态保留。

## DEC-010 — REFINE + IMPLEMENT
H-007对本轮预期回升为CURRENT_EVIDENCE_AGAINST但约束作用不可识别，H-008 INSUFFICIENT_EVIDENCE。暂停额外微调，转向已有检查点同预算评价和BN校准先导；保留ROUND-009所有结果及梯度检查时序偏差。

## DEC-011 — PIVOT + IMPLEMENT
H-009 CURRENT_EVIDENCE_SUPPORTS限定原诊断条件；H-010 CURRENT_EVIDENCE_AGAINST限定主要20%预算与C0比较，C3比较证据不足。停止后训练阈值/BN校准作为主线，转向阈值幅值及重置路径代理梯度；统一缩放保留为强对照。

## DEC-012 — REPAIR + REPLICATE
H-011 INSUFFICIENT_EVIDENCE：原筛选失败且最终模型不可分辨，不推广为代理梯度普遍无效。先修复共享阈值梯度干预尺度；仅当EXP-012的全部工程门槛通过后开展性能实验。

## DEC-013 — STOP + PIVOT (ROUND-013)
User approved stopping the current ARTG optimization direction while retaining H-012's shared-threshold gradient-scale finding. Investigate true structured internal-channel pruning; do not infer novelty for direct removal loss. No training or official test set in EXP-014.

DEC-013 execution evidence: EXP-014 completed; both prespecified screening gates were not met. This is an observation for theory review, not an autonomous new research decision.

## DEC-014 — STOP + PIVOT
User and theory review formally stop the current AT-LIF local improvement mainline while preserving all historical experiments; prioritize UniSparse stage-0 engineering admission, with no claim of reproduction or originality. H-014/H-015 are CONTRADICTED only within EXP-014's fixed conditions; H-012 remains SUPPORTED, H-013 INSUFFICIENT_EVIDENCE.

DEC-014 stage-0 evidence: EXP-015 partial; admission blocked pending isolated compatible CuPy environment and required controlled tests. This records the gate outcome, not a new research direction decision.

## DEC-015 — REPAIR + CONTINUE
User approved isolated CuPy environment repair, official backend admission tests and synthetic effective-connectivity activity-cost checks, without formal ECAS training, 300-epoch training or test set access. EXP-015 remains PARTIAL as a historical record.

DEC-015 execution evidence: engineering admission passed under the tested environment and limited examples; formal full-training approval remains PENDING and paper nonzero λ remains unresolved. H-016 synthetic mechanism passed pending theory; this is not a new algorithm or autonomous research-direction decision.

## DEC-016 — CONTINUE + BASELINE
User approved EXP-017 U0/U1/U2, each 20 epochs of 300-epoch cosine schedule, with locked R68 reference if official λ is unverified. Preserve author updates including AMP-skip manual penalty; no official test, extra candidates, >20 epoch training or ECAS.

DEC-016 execution: EXP-017 completed; U1/U2 both passed fixed pilot screening, U2 ranked first by prespecified lower-SOP criterion. This is an exploratory recommendation for later approval, not authorization for additional training.

## DEC-017 — CONTINUE + BASELINE
User selected U2 as controlled combined-sparsity baseline, U0 as dense-control continuation, and U1 as backup only. Approved up to 300 epochs with mandatory 60/100/200 gates and R69–R73 cost attribution, within one RTX4090/8-hour and CPU16-hour cap. No U1 continuation, official test, ECAS, retraining, search or change to author updates.


## DEC-018 — IMPLEMENT + CONTINUE
End UniSparse controlled-baseline preparation; test MP-EFR as a specific unproven incremental mechanism, without claims of novelty. EXP-020 only; see ROUND-018-TASK.md. Preserve all ROUND-017 evidence and fixed H-018 gate.


## DEC-019 — REFINE + CONTINUE
Keep MP-EFR unchanged; perform finite calibration-only G0/G2 budget matching around fixed G1 q=.04. Preserve EXP-020 failure and do not infer training effects.
