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
