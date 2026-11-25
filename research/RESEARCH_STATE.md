# ROUND-003 科研状态

DEC-003 REPAIR → CONTINUE，理论 v0.1.1；Q-001 主线，Q-002 竞争解释，H-001/H-002 UNTESTED。

EXP-001 COMPLETED；EXP-002/003 BLOCKED；EXP-004 IMPLEMENTED_NOT_RUN。已生成 CIFAR-10 训练来源 45000/4500/500 分层索引及 256 诊断样本。独立训练验证入口、状态隔离 A/B/C CPU 对照及 CPU 恢复训练测试通过；A/C 阈值差 0，B/A 为 0.000124812126。原本训练入口不再作为受控版本使用。

GPU benchmark RUN-002（代码 `4d8272eee513ae5e303d9a8cb3cc97d4aec82b68`）完成，单卡 4090、约 4.46 GiB allocated、训练均值 0.198 秒/batch，完整训练粗估约 4.28 小时；首次 RUN-001 设备编号失败。旧代码设种子和评估保存全部 CUDA RNG，可能触及其他可见 GPU，已修复，仅在所选 GPU 上执行了训练计算；修复后的 GPU 路径仍需检查。未启动完整训练，未将短基准作为假设证据。

最终训练入口代码 `7afbeea1afbe80fccff16636d13457e88153102e`；需理论审阅 AMP 跳步和 GPU 状态隔离，并由用户单独批准正式训练资源及终止条件。参见 ../experiments/EXP-004/REPORT.md。ROUND-002 结果提交 `7b9372b558cbf32d07847cea2b4008e6f9867cff`。
