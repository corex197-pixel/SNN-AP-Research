# ROUND-003 科研状态

DEC-003 REPAIR → CONTINUE，理论 v0.1.1；Q-001 主线，Q-002 竞争解释，H-001/H-002 UNTESTED。

EXP-001 COMPLETED；EXP-002/003 BLOCKED；EXP-004 IMPLEMENTED_NOT_RUN。已生成 CIFAR-10 训练来源 45000/4500/500 分层索引及 256 诊断样本。独立训练验证入口、状态隔离 A/B/C CPU 对照及 CPU 恢复训练测试通过；A/C 阈值差 0，B/A 为 0.000124812126。原本训练入口不再作为受控版本使用。

GPU benchmark RUN-002（代码 `4d8272eee513ae5e303d9a8cb3cc97d4aec82b68`）完成，单卡 4090、约 4.46 GiB allocated、训练均值 0.198 秒/batch，完整训练粗估约 4.28 小时；首次 RUN-001 设备编号失败。旧代码设种子和评估保存全部 CUDA RNG，可能触及其他可见 GPU，已修复，仅在所选 GPU 上执行了训练计算；修复后的 GPU 路径仍需检查。未启动完整训练，未将短基准作为假设证据。

最终训练入口代码 `7afbeea1afbe80fccff16636d13457e88153102e`；需理论审阅 AMP 跳步和 GPU 状态隔离，并由用户单独批准正式训练资源及终止条件。参见 ../experiments/EXP-004/REPORT.md。ROUND-002 结果提交 `7b9372b558cbf32d07847cea2b4008e6f9867cff`。


## ROUND-004 — 前置检查受阻

用户批准单 RTX 4090、含前置检查的 8 小时上限以及 EXP-004-RUN-001 一次 200 epoch 训练。协议与代码见 ROUND-004-TASK.md、experiments/EXP-004/round004/。前置检查 PREFLIGHT-001 的单卡、划分、GPU A/C、检查点一致性通过，但 AMP 受控测试的活动量为 0，不能充分验证手动更新；PREFLIGHT-002 启动时 GPU 已被其他进程占用，可用显存低于 8 GiB，故中止。当前无实际空闲 GPU；正式训练未启动。GPU 预算记录见 round004/budget.json，不延长既定绝对截止时间。H-001/H-002 UNTESTED。


## ROUND-004 正式运行审计 — PARTIAL

EXP-004-RUN-002 仅完成 37/200 轮；训练与监护进程均已退出，原状态仍为 RUNNING，确切退出时间和原因未知。最佳验证第 34 轮 accuracy 0.814、loss 0.5378644655、FR_global 0.0576531808、FR_author 0.0489380468。最新和最佳检查点哈希与指标记录一致，已在 CPU 验证可读及参数有限；续训未测试。无最终 SOP 数据，未使用正式测试集。H-001/H-002 UNTESTED，EXP-002/003 BLOCKED。详见 ../experiments/EXP-004/REPORT.md 及 audit_summary.json。


ROUND-005 预注册：用户已批准单卡7小时；恢复前测试和续训结果以 round005/SUPERVISOR-001/status.json 及独立恢复段实际记录为准。父运行永久保留 PARTIAL，不重置随机种子或200轮调度器。


## ROUND-005 恢复运行

父运行 EXP-004-RUN-002 保留 PARTIAL；恢复段 EXP-004-RUN-003 状态 COMPLETED，最近完成 Epoch 200，全程最佳 Epoch 194，最佳验证准确率 0.9171111111111111。执行代码 `03def08d9fab5c37968a2a2eaa37675666cb4110`；详细退出记录、GPU 预算及检查点哈希见 round005/SUPERVISOR-001/delivery.json；正式运行数据见 runs/EXP-004-RUN-003/。最终固定验证 已产生 final_summary.json。不使用正式测试集，H-001/H-002 保持 UNTESTED。


## ROUND-006 状态

EXP-002 COMPLETED：中部模块alpha=1.5，R_FR=0.597728，mean E=0.026041，95% CI [0.02574658759203404, 0.026323316503716793]；依据预注册阈值该条件对具有实际幅度的H-001提供反对证据。任务损失变化 -0.024406，CI [-0.09605079152643584, 0.0389564814817811]，EXP-003 门槛未达到，BLOCKED。H-001/H-002 科学状态仍 UNTESTED，等待理论研究员审查；后续补充层为探索性，不能替换主要条件。未使用测试集或训练。详见 ../experiments/EXP-002/REPORT.md。


## ROUND-007 科研状态（覆盖旧状态中相同假设的过时标记）

原H-001 CURRENT_EVIDENCE_AGAINST，范围限原中部主要条件；H-002 UNTESTED，EXP-002 COMPLETED，EXP-003 BLOCKED；EXP-004 COMPLETED。

EXP-005 COMPLETED：244个剩余诊断样本，前部 alpha=1.5，H-004预设门槛达到；R_FR=0.444135，E=0.168514，损失变化=0.144484。EXP-006 COMPLETED：R_E=0.667882，R_loss=0.172414，预设H-005门槛及G4特异性判定均未达到，不能排除效应，证据不足。H-004/H-005 的正式科学状态等待理论研究员审查。诊断池500张现均用于机制分析，后续不可再视为独立最终评价集。详见 EXP-005/006 报告。


## ROUND-008 科研状态

DEC-008 PIVOT + IMPLEMENT；原幅值补偿主线暂停，H-001 CURRENT_EVIDENCE_AGAINST（原限定条件）、H-004 CURRENT_EVIDENCE_SUPPORTS（前部限定条件）、H-005 INSUFFICIENT_EVIDENCE，保留历史原始证据。EXP-007-RUN-001 INVALID（B2循环顺序实现错误），修复后的RUN-002 COMPLETED。M1验证SOP降幅 10.4711%，准确率 89.9333%，相对B0下降 1.7778个百分点；预设筛选信号未达到。M1与B2预算匹配但未展现明确优势，B1预算不匹配。H-006保留UNTESTED待理论审查。详见 ../experiments/EXP-007/REPORT.md。

## ROUND-009 预注册
DEC-009 PIVOT + IMPLEMENT；H-006 INSUFFICIENT_EVIDENCE。EXP-008 PLANNED，H-007/H-008 UNTESTED。原动态贪心搜索停止；固定预算内开展四组微调，详见ROUND-009-TASK.md。

EXP-008 COMPLETED；H-007预设效果门槛=False，H-008预设效果门槛=False，整体筛选信号=False。正式科学假设状态仍UNTESTED待理论研究员审查；详情见../experiments/EXP-008/REPORT.md。

## ROUND-010 预注册
DEC-010 REFINE + IMPLEMENT；暂缓额外梯度训练，EXP-009和EXP-010 PLANNED；H-007 CURRENT_EVIDENCE_AGAINST（限定回升预测，约束效应不可识别），H-008 INSUFFICIENT_EVIDENCE，H-009/H-010 UNTESTED。

EXP-009、EXP-010 COMPLETED。20%目标下C1/C2不优于C0；P0 batch BN相对eval的SOP变化为 -27.4200%，且重复稳定，达到H-009预设探索性幅度。H-010筛选门槛未达到。H-009/H-010正式状态仍UNTESTED，待理论研究员审查；详见EXP-009/010报告。

## ROUND-011 预注册
DEC-011 PIVOT + IMPLEMENT；H-009 CURRENT_EVIDENCE_SUPPORTS（指定模式条件）、H-010 CURRENT_EVIDENCE_AGAINST（指定20%预算相对C0），H-011 UNTESTED。EXP-011 PLANNED；严格先完成前向及梯度前置检查，三组最多各5轮。

EXP-011 COMPLETED；H-011预设筛选=False。G0前四轮后因发现M4混合精度问题中断并恢复，失败记录和两个代码版本均已保存；全部真实结果详见../experiments/EXP-011/REPORT.md。H-011正式状态仍UNTESTED，待理论研究员审查。

## ROUND-012 预注册
DEC-012 REPAIR + REPLICATE；H-011 INSUFFICIENT_EVIDENCE，EXP-012 PLANNED、EXP-013 BLOCKED等待进入门槛；H-012/H-013 UNTESTED。保留ROUND-011前置异常、中断及恢复记录。

EXP-012 COMPLETED，H-012工程进入门槛通过；EXP-013 COMPLETED，H-013预设筛选=False。H-012/H-013正式科学状态仍为UNTESTED，待理论研究员审查；见../experiments/EXP-012/REPORT.md和../experiments/EXP-013/REPORT.md。

## ROUND-013 initiated
DEC-013 approved STOP + PIVOT away from ARTG. EXP-012 and EXP-013 remain COMPLETED, H-012 SUPPORTED within its tested scope and H-013 INSUFFICIENT_EVIDENCE. EXP-014 is preregistered; H-014/H-015 remain UNTESTED pending theoretical review. Index split, code, and protocol committed before formal evaluation.

### ROUND-013 result
EXP-014-RUN-001 COMPLETED: all 224 A and B single-channel measurements and five 4500-sample validation groups. Synthetic and four real 32-sample equivalence checks passed. All compact groups removed 96,880 parameters (8.93%). H-014 and H-015 preregistered screening gates not met; both remain UNTESTED pending theory review. P-Loss worse than FR/L1, and its SOP savings is >2 percentage points below FR. No new training or official test. See experiments/EXP-014/REPORT.md.

## ROUND-014 — DEC-014
Stop active AT-LIF local-improvement route and preserve its 13 rounds. New UniSparse Q-010 is at stage-0 engineering admission, not reproduced. EXP-014 remains COMPLETED; H-014/H-015 CONTRADICTED only under the prespecified EXP-014 conditions. EXP-015 is preregistered; no additional performance hypothesis is asserted.

EXP-015-RUN-001 PARTIAL; UniSparse admission BLOCKED: official CuPy backend missing. CPU weight/penalty and binary SOP synthetic tests passed, train-only loader smoke passed, but CuPy equivalence, GPU forward/backward and benchmark are untested. GPU use 0. No 300-epoch cost estimate or pretrained weights verified. See ../experiments/EXP-015/REPORT.md.

## ROUND-015 preregistration
DEC-015 REPAIR + CONTINUE: retain UniSparse as priority candidate; EXP-015 remains PARTIAL. EXP-016 tests official CuPy backend and effective-connectivity accounting. H-016 and H-017 remain UNTESTED pending researcher review. No ECAS training or official test data.

## ROUND-015 results
EXP-016 COMPLETED. CuPy 13.5.1 installed in the dedicated inherited-package venv; first official backend run failed on np.int and was preserved, isolated compatibility patch followed by backend success. CifarNet GPU forward/backward and nonzero penalty smoke passed. Synthetic H-016 mechanism gate passed; scientific state awaits theoretical review; H-017 UNTESTED. Engineering admission is ENGINEERING_ADMITTED within tested conditions; formal 300-epoch training PENDING, formal nonzero λ UNKNOWN. EXP-015 remains PARTIAL. The paper explicitly described initial-structure approximation; no novelty or trained-network performance claim. See ../experiments/EXP-016/REPORT.md.

## ROUND-016 initiated
DEC-016 approved controlled UniSparse working-point pilot EXP-017; EXP-015 remains PARTIAL, EXP-016 COMPLETED, engineering admitted only. H-016 PARTIALLY_SUPPORTED only for synthetic fixed conditions; H-017 UNTESTED. Full reproduction and full training remain PENDING. Preflight and configuration lock precede three independent 20-epoch runs.

### ROUND-016 result
EXP-017 COMPLETED: U0/U1/U2 each 20 epochs, same seed, original UniSparse updates, 300-epoch scheduler; no official test. Both U1 and U2 pass preregistered screening, U2 recommended by lowest sample-weighted SOP; formal 300-epoch training remains PENDING separate approval. H-016 PARTIALLY_SUPPORTED only for prior synthetic mechanism; H-017 UNTESTED. Data and recovery audits in ../experiments/EXP-017/.

## ROUND-017 initiated
DEC-017 approved U0/U2 continuation from epoch20, mandatory 60/100/200 gates, epoch20/60/100/200/300 counterfactual cost decomposition; U1 remains backup and EXP-017 remains completed. H-016 PARTIALLY_SUPPORTED, H-017/H-018 UNTESTED. Original UniSparse algorithm, fixed data and official-test isolation preserved; complete reproduction not claimed.

ROUND-017 delivery: EXP-018 COMPLETED at {'U0': 300, 'U2': 300}; EXP-019 COMPLETED at epochs [20, 60, 100, 200, 300]. H-016 PARTIALLY_SUPPORTED; H-017/H-018 UNTESTED pending theory review. See EXP-018/019 reports.

ROUND-017 final delivery verification: original delivery supervisor exited for unknown reason; replacement completed delivery. Epochs 21–300 are continuous for both groups, sample pair audits pass, all parent and epoch-300 checkpoint hashes match, and all 1,120 validation shard hashes match. The epoch-60 H-018 amplitude gate remains unmet; H-017/H-018 remain UNTESTED pending theoretical review. Detailed CPU-only audit: ../experiments/EXP-018/final_delivery_audit.json.


ROUND-018 DEC-018 IMPLEMENT + CONTINUE: EXP-020 PLANNED, fixed U2 epoch300 and R74–R80, no training or official test. H-016 SUPPORTED scoped mismatch; H-018 PARTIALLY_SUPPORTED; H-017/H-019 UNTESTED. Stage gates and resource authorization in ROUND-018-TASK.md.

EXP-020-RUN-001 PARTIAL: real local replay, R78 and SOP reference checks passed; B0 channel reporting failed due to tensor-to-scalar indexing error. Preserved raw files; RUN-002 corrects reporting and deducts RUN-001 resource use.

ROUND-018 EXP-020 COMPLETED: q=.02 H-019 direction passed; G1 matched q=.04, G0/G2 unmatched at fixed scales, so overall training progression gate failed and paired matched-control CI unavailable. H-017/H-019 remain UNTESTED pending theory review. RUN-001 failure and locked-condition FR supplement preserved; cumulative GPU wall and CPU in EXP-020/resource_usage.csv.


ROUND-019 DEC-019 REFINE + CONTINUE; EXP-020 COMPLETED with original progression gate failed; EXP-021 PLANNED with fixed G1 q=.04, finite G0/G2 cost-only matching and locked validation. H-019 SUPPORTED scoped; H-017 INSUFFICIENT_EVIDENCE, training impact untested.

ROUND-019 EXP-021 COMPLETED. Budget matching succeeded; G1−G0 accuracy −0.04444 pp (gate fails), G1−G2 +0.53333 pp (gate passes). Overall progression failed; H-017 INSUFFICIENT_EVIDENCE pending theory review, H-019 scoped SUPPORTED per theory decision. EXP-020 original gate remains failed. See ../experiments/EXP-021/REPORT.md.


ROUND-020 DEC-020 REFINE; EXP-020/021 COMPLETED with original progression gates failed. EXP-022 PLANNED; first test real full-network task gradient conflict; P1 mechanism and matched validation only if preregistered gate passes. H-020/H-021 UNTESTED.

ROUND-020 EXP-022 COMPLETED: true task/activity conflict G1 conv1.norm cosine −0.155419; P0/P1 matched budget, but P1 failed all three accuracy advantage gates versus G0/G1/P0. No continuation of current MP-EFR/TP-EFR training mainline is recommended under preregistered rule. H-020/H-021 remain UNTESTED pending theory review. EXP-020/021 failure records preserved. See ../experiments/EXP-022/REPORT.md.


ROUND-021 DEC-021 STOP + PIVOT: MP-EFR/TP-EFR training mainline stopped, controlled UniSparse retained; EXP-020/021/022 COMPLETED with original negative gates preserved. EXP-023 PLANNED; H-022 gate determines whether W0/W1/W2 101–110 training is permitted. H-020 SUPPORTED scoped; H-021 CONTRADICTED scoped; H-022/H-023 UNTESTED.

ROUND-021 EXP-023 delivery: status COMPLETED, completed {'W0': 110, 'W1': 110, 'W2': 110}; H-022 admission True, H-023 advancement False. H-022/H-023 remain UNTESTED pending theory review; prior negative evidence retained. See ../experiments/EXP-023/REPORT.md.

ROUND-021 protocol deviation: pre-training unequal-tail unit test was mislabeled (1+1); genuine 2+1 test passed only post-run. EXP-023 executed to completion; original full correctness admission was not entirely verified at start. H-023 paths both failed; theory review must account for this execution deviation.


ROUND-022 DEC-022 STOP + PIVOT: stop fixed-layer SEAWP allocation; retain controlled UniSparse and all EXP-023 negative results and admission deviation. EXP-024 PLANNED; fixed 25% fc1 hidden-neuron structural pruning, exact reconstruction and machine-verifiable admission precede selection/evaluation. H-024/H-025 UNTESTED.

ROUND-022 EXP-024 delivery: PARTIAL; admission RUNNING; progression NA. H-024/H-025 UNTESTED pending researcher review. EXP-023 admission deviation remains preserved. See ../experiments/EXP-024/REPORT.md.

EXP-024-RUN-001 PARTIAL: R103 error 1.74997e-5 exceeded 1e-5; no selection or validation. Preserved run; RUN-002 checks strict FP32 by disabling TF32, retains tolerance and counts previous budget.

ROUND-022 EXP-024 delivery: COMPLETED; admission PASSED; progression False. H-024/H-025 UNTESTED pending researcher review. EXP-023 admission deviation remains preserved. See ../experiments/EXP-024/REPORT.md.


ROUND-023 DEC-023 STOP + REASSESS: ESC-JP/M5 current formal route STOPPED; UniSparse remains current Controlled Baseline; all candidates UNDER_REVIEW, none ENGINEERING_ADMITTED. EXP-024 COMPLETED and progression FAILED: H-024 SUPPORTED scoped and H-025 PARTIALLY_SUPPORTED (S3 only; gates vs S1/S2 failed). RUN-001 error and 45 strict/default FP32 prediction differences remain preserved; EXP-023 pre-run admission deviation remains. EXP-025 static review COMPLETED; C-DPSS priority conditional on license, repairing CIFAR test-as-validation leakage and remaining checks. No GPU used. See ../experiments/EXP-025/REPORT.md.


ROUND-024 DEC-024: retain UniSparse; C-DPSS/SCA/Dynamic and Spik4lite UNDER_REVIEW; M5 remains STOPPED, EXP-024 progression FAILED and EXP-025 CPU accounting gap retained. EXP-026 ongoing: 50% primary, 75% exploratory, fixed-support ridge, no SGD/test; H-026/H-027 UNTESTED. See ../experiments/EXP-026/status.json.


ROUND-024 EXP-026 process COMPLETED, formal execution INVALID: baseline reconstructed; preselection check coverage failed and automatically marked PASSED; exploratory evidence retained; H-026/H-027 remain UNTESTED pending theoretical review. No official test or SGD. See ../experiments/EXP-026/REPORT.md.

ROUND-025 DEC-025: EXP-026 remains INVALID and its H-026/H-027 evidence INVALID_EVIDENCE; independent EXP-027 planned with actual-object gate and unchanged candidates. UniSparse remains Controlled Baseline; no switch or new method.
