# EXP-029 部分执行与准入偏差报告

正式状态 PARTIAL，训练证据 INVALID_EVIDENCE；未启动终点准入、未创建正式验证加载器或评价。原训练锁 `852a8b43f148bccfcdd718d1be3cb2933f157c64` 在必要覆盖不足时被错误接受，不能追认。

## 已观测的真实逐时间步预检
固定 D_grad=512；权重非均匀样本比例 1.000000，循环错位后 2037 个元素不同；权重范围 [0.5, 2.5]，时间平均最大偏差 1.19e-07。
真实 R127/R128/R129 归一化最大误差为 2.76e-07 / 4.41e-07 / 6.8e-07，真实保留神经元脉冲不一致数 0。KD FP32 对独立 FP64 数值误差、权重边界和负面入口测试见 formula_tests.json 和 admission_negative_tests.json。

## 准入覆盖偏差
原预检包含真实网络 R127–R129 检查及权重人造边界案例，但**未单独执行并记录预注册的人造矩阵 R127–R129 检查**。此外，真实临时模型恢复检查比较了模型、优化器状态条目数、调度器和 scaler，但未比较所有优化器动量张量。原必需测试清单没有识别这两项缺失，导致训练前错误通过。发现后对本任务训练子进程发送 SIGTERM；日志中的 watchdog signal 15 代表人工停止，不是预算耗尽。
事后 CPU 人造矩阵六项误差均为零；新版动量状态比较器的人工正反测试通过。这些结果不能事后恢复原预评价准入。修复后的未来清单要求 `synthetic_readout_127_129` 和 `optimizer_recovery_exact`，本轮未重新运行完整 GPU 预检。

## 训练实际状态与检查点
| 组 | 已完成 Epoch | 已完成 Epoch 的成功更新 | AMP 跳步 | 检查点 |
|---|---:|---:|---:|---|
| K0 | 5 | 875 | 0 | /home/hym/keyan1/experiments/EXP-029/models/K0/epoch-005-recovery.pth |
| K1 | 5 | 875 | 0 | /home/hym/keyan1/experiments/EXP-029/models/K1/epoch-005-recovery.pth |
| K2 | 5 | 875 | 0 | /home/hym/keyan1/experiments/EXP-029/models/K2/epoch-005-recovery.pth |
| K3 | 1 | 175 | 0 | /home/hym/keyan1/experiments/EXP-029/models/K3/epoch-001-recovery.pth |
| M6 | 0 | 0 | 0 | NA |
K3 第 2 轮开始后中断，未保存该轮恢复状态，精确已尝试更新次数未记录。K0/K1/K2 第 5 轮推理检查点和 K3 第 1 轮恢复检查点存在、哈希与 CPU 加载检查通过；这不意味着训练准入有效。所有路径和 SHA256 见 partial_checkpoint_manifest.json。

## 未完成的比较与科研状态
未开展七组验证集评价、终点准入、时间 logits、K0 历史 F0 预测重建、配对统计或性能门槛判定；相关指标均为 NA。H-030/H-031/H-032 保持 UNTESTED。任何后续新训练必须有独立预检、可核验锁和资源授权；不得通过事后补测恢复本次训练的合规性。

## 资源及既有证据
已结束 GPU 进程墙钟累计 464.96s；CPU 已计入 1017.41s（含启动前 120s 保守预留），CPU 事后整理另计于 resource_usage.csv；GPU 墙钟包含 CPU 和等待，并非内核时间。
首次资源基准将 K0 的一次性冷启动约 3.75s 当作每批，触发停止；第二次预检使用三个实际连续批次的最大耗时重估，记录两次测量与各自 Run ID。两次预检及失败均保留。没有读取正式测试集、未训练全模型。先前 EXP-026 INVALID、EXP-027 VALID、EXP-028 修复后的评价 VALID 均不变。
