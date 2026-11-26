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
