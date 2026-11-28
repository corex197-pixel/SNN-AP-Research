# ROUND-001 预注册方案

完整协议、指标、阈值及预算见 [ROUND-001-TASK.md](ROUND-001-TASK.md)，原样保存于运行前。EXP-000 审查；EXP-001 CPU 合成机制及实际 ATLIF 单元测试；EXP-002 仅在可靠权重、数据及其他前提齐备后执行；EXP-003 仅在 EXP-002 所有预设门槛通过后执行。

资源估计：CPU 少于 10 分钟，GPU 0；硬预算 CPU 累计 30 分钟、单 GPU 累计 1 小时；不训练、不下载。R1/R2 相对误差容差 float64 1e-10，float32 1e-5。
ROUND-002 修复与冒烟协议见 ROUND-002-TASK.md；EXP-004 尚未正式训练，安全入口、数据划分和资源预算待解决，见 ../experiments/EXP-004/REPORT.md。

ROUND-003 协议见 ROUND-003-TASK.md；用户只授权 EXP-004-BENCH-01 单卡测量，不授权完整训练。已固定分层索引及独立验证入口；完成正式 GPU 路径审查与资源批准后方可启动训练。

ROUND-004 预注册任务与用户单卡八小时批准见 ROUND-004-TASK.md；因缺少实际空闲 GPU，非零活动 AMP 前置验证和正式训练尚未完成。

ROUND-004 EXP-004-RUN-002 运行中断于 37 个已完成 Epoch；GPU 预算已到期，本次仅对已有记录 CPU 审计，后续 SOP 评估或续训需单独授权。

ROUND-005 经用户授权，从第37轮最新检查点恢复第38—200轮，单张 GPU 总预算 7 小时；恢复测试、正式续训及最终验证包含在内。历史 37 轮与最佳模型保持不变，不额外训练。详见 ROUND-005-TASK.md。

ROUND-006 预注册：固定主要模块 conv3_x.1.residual_function.2、alpha=1.5；原 R4—R12 和预设进入条件保持不变，v0.1.2 增加辅助 R14。固定诊断池 256 样本，种子2026；按配对样本 Bootstrap 2000次 seed2027。详见 ROUND-006-TASK.md。

ROUND-007预注册：独立训练来源剩余244个索引，conv1.2，alpha 1.0与1.5；H-004的FR/E/损失及完整性门槛均须满足，且裁剪后的可达恢复比例>=0.5才能执行EXP-006；G4为样本级统一幅值竞争对照。原H-001结论和门槛保留。

ROUND-008预注册EXP-007：固定训练集来源分层500校准图、seed2029；B0/B1/B2/M1按ROUND-008协议，仅校准集选择阈值，锁定后4500验证集评价，主目标SOP节省10%，12动作上限；不使用诊断池或测试集。

EXP-007-RUN-001已停止并标记INVALID：B2未按固定排序循环。修复后的RUN-002预先规定B2接受动作后从排序中下一模块继续；初始无效而无分数模块置于排序末尾按模块顺序排列，只有实际边际SOP收益达到原门槛时可接受。保留RUN-001完整记录及资源消耗，RUN-002预算扣除已用时间。

## ROUND-009 / EXP-008 预注册
固定同一194轮检查点及45000/4500划分；B0/P0推理对照、F0/F1/F2/M2各5轮独立同初始化微调，1.25倍固定阈值，seed=2030。固定C0、公式、学习率、无测试集、预算和预设H-007/H-008门槛详见ROUND-009-TASK.md。仅第5轮为主结果；运行前检查代理计数误差≤1e-5与梯度。

## ROUND-010 预注册
EXP-009 C0/C1/C2固定12倍率、20%/30%目标及最多3次跨越区间中点；EXP-010包含同输入BN模式对照、普通Adaptive BN和按执行顺序的总体方差逐层校准。候选仅在固定500校准样本选择，全部配置锁定后才使用4500验证样本；FP32，无训练、无诊断池及测试集。完整公式、预算和门槛见ROUND-010-TASK.md。

## ROUND-011 / EXP-011 预注册
G0原代码、G1公式13局部因子对齐、M4幅值和重置独立路径；同一第194轮模型、阈值初始化×1.25、sp=0.002、3组各5轮。必须先通过人工单双路径梯度、多时间步、真实网络与AMP前置测试；所有最终第5轮检查点独立以其学习阈值结构全局倍率校准，锁定后评价固定验证集。详情见ROUND-011-TASK.md。

## ROUND-012 预注册
EXP-012 D0/G0-S/G1-S/M4-S每组最多20次batch、16次成功更新；同时满足所有数学、有限性、非零更新模块数和8 ULP向量分离门槛才运行EXP-013三组各5轮。固定原检查点及划分，动态N按实际batch计算，后缩放只作用阈值任务梯度。所有公式、门槛与固定评价详见ROUND-012-TASK.md。

## EXP-014 preregistration
Full fixed protocol: research/ROUND-013-TASK.md. Three preselected internal ATLIF modules, 25% per layer (8/16/32); baseline and independent Random, L1, firing-rate and actual loss-ranked compact models. A determines choices; disjoint B tests ranking stability only. All selections locked before 4500 validation samples. Equivalence tolerance 1e-5. Paired sample bootstrap 2000, seed 2027. No gradient training, BN recalibration, official test or extra pruning-rate search.

## ROUND-014 EXP-015 preregistration
Official UniSparse SHA b5b3da4fcd2e7abcd2c20c93d41cf63ef3201daa, MIT. Independent environment and external controlled adapters. CPU synthetic correctness and author SOP audit first; GPU short benchmark only after official CuPy forward/backward and sparse-state prerequisites, single RTX4090, at most 10 warmup plus 30 measured batches, batch256, T4, seed12450, <=600 GPU seconds, <=2700 cumulative CPU seconds. No test set, full epoch, AT-LIF tuning, or new algorithm. Full protocol: ROUND-014-TASK.md.

## ROUND-015 / EXP-016 preregistration
See ROUND-015-TASK.md. Upstream b5b3da4fcd2e7abcd2c20c93d41cf63ef3201daa; RTX4090 one GPU <=900 seconds; CPU <=2700 seconds; downloads <=200MB; dedicated dependency disk <=2GiB. CPU R57–R61 mechanism tests, official CuPy/Torch comparisons, full-model checks and only then bounded short benchmark. No 300-epoch training or official test set. All failures and skipped tests remain recorded.

## EXP-017 / ROUND-016 preregistration
Official CifarNet T4, seed12450, batch256, training 45000/validation4500 from train-only data, 175 train batches with drop_last, no official test. U0=0, U1=λ_ref, U2=3λ_ref if author value unknown, λ fixed prior to formal training. Each independently initialized for 20 epochs with 300-epoch cosine; fixed epoch 20 primary comparison, screening criteria and complete budget in ROUND-016-TASK.md. GPU wall <=4500s; CPU cumulative <=10800s. Save metrics, evaluation isolation, full resumable checkpoints, samples and finite-state audit.

## EXP-018/019 — ROUND-017 preregistration
Exact parent epoch20 U0/U2 SHA256 and fixed data/lambdas are in ROUND-017-TASK.md. Preflight exact state and paired update recovery, then epoch20 cost attribution; paired stage endpoints 60,100,200,300 with gates accuracy drop<=5pp, SOP savings>=5%, all penalized densities>=1%, finite and restorable, equal epochs. Four-mask costs use same actual monitored binary inputs, author scope, 4500 train-origin validation and sample-weighted aggregation. Epoch60 H-018 primary; synthetic accounting is not training causality. User budget GPU 28800 wall seconds and CPU57600 cumulative seconds.


## ROUND-018 EXP-020
Preregistered protocol ROUND-018-TASK.md. Fixed 512 D_grad and 512 D_cal, seed 20261128, disjoint from validation and old diagnostic pool. G0/G1/G2, rho=.25, q=[0,.0025,.005,.01,.02,.04,.08,.16], primary q=.02, SOP target 5% and admissible 4–6%; validation only after lock. GPU≤1800s CPU≤3600s. No training or official test.


## ROUND-019 EXP-021
Fixed parent, samples, rho and permutations from EXP-020. G1 q=.04; G0/G2 initial [0.04,0.08], R82 then ≤4 bisections, abs calibration reduction mismatch≤0.001 and range 4–6%; commit lock before validation. Fixed performance gates unchanged; GPU≤900s, CPU≤1800s. Full protocol ROUND-019-TASK.md.


## ROUND-020 EXP-022
Fixed parent, D_grad/D_cal, BN target paths, rho=.25 and G0/G1 q from EXP-021. H-020 requires at least one G1 layer cosine <−1e−6; if absent stop protected performance. Otherwise same-q mechanism q=.02, fixed finite P0/P1 SOP matching and preregistered 2×2 tests; GPU≤900s, CPU≤1800s. Full protocol ROUND-020-TASK.md.


## ROUND-021 EXP-023
Fixed U2 epoch100 parent, split and 512+512 exposure samples, alpha=.25, unmodified activity regularizer. H-022 requires both D_grad and D_cal A_p≥.20 with same min/max layer in conv1.conv/conv2/conv3.conv. If passed and tests passed: independent W0/W1/W2 epochs101–110, fixed endpoint110; GPU≤2700s CPU≤5400s. Full protocol ROUND-021-TASK.md.


## ROUND-022 EXP-024
Fixed U2 epoch300 parent, D_grad/D_cal/4500 validation, original FP32 model and FP64 algebra; B0/S0/S1/S2/S3/M5, 512 of 2048 deleted, M5 block=16, seed20261128. Actual admission tests with inputs/batch sizes/assertions/results precede selection; selection commit before D_cal/validation. GPU≤1200s, CPU≤2700s. Full ROUND-022-TASK.md.


ROUND-023 EXP-025 COMPLETED as static audit; three candidates UNDER_REVIEW, not admitted. Next-stage conditional engineering proposal in ../experiments/EXP-025/next_stage_proposal.md requires independent user authorization; no experiment started.
