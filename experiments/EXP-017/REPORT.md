# EXP-017 — UniSparse 受控训练工作点先导

执行状态：COMPLETED。运行代码：`c718afecce6c65bcccbe8ba2b3e23ba07b2a7294`；前置检查：`2d26c9f50ca75adf24ad79b42e6a54c94c0cb7ec`。H-016 限定 PARTIALLY_SUPPORTED；H-017 UNTESTED。

非零论文 λ 未确认；R68 λ_ref=2.94470291242e-08。U0=0，U1=λ_ref，U2=3λ_ref。训练与验证仅使用 CIFAR-10 训练来源固定划分，无官方测试。

| 组 | 已完成轮数 | 第20轮准确率 | 第20轮损失 | 加权 FR | 加权 SOP/样本 | 受惩罚层稀疏率 | AMP 跳步 | 手动更新 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U0 | 20 | 0.832 | 0.49738044 | 0.035879794 | 2.7355199e+08 | 8.1725271e-08 | 0 | 0 |
| U1 | 20 | 0.82133333 | 0.52431025 | 0.036588846 | 2.298285e+08 | 0.01682391 | 0 | 3500 |
| U2 | 20 | 0.81622222 | 0.54396582 | 0.037342085 | 1.5774796e+08 | 0.045071705 | 0 | 3500 |

预注册工作点选择：{"conditions": {"U1": {"penalized_sparsity_ge_1pct": true, "sop_savings_ge_5pct": true, "accuracy_drop_le_5pp": true, "each_penalized_layer_density_ge_1pct": true, "finite_and_checkpoint": true, "passed": true}, "U2": {"penalized_sparsity_ge_1pct": true, "sop_savings_ge_5pct": true, "accuracy_drop_le_5pp": true, "each_penalized_layer_density_ge_1pct": true, "finite_and_checkpoint": true, "passed": true}}, "recommended": "U2", "reason": "pre-registered SOP then accuracy then U1 tie-break; single seed exploratory", "comparison_epoch": 20}。不以中途最佳轮次替代固定第20轮。

资源状态：{"status": "COMPLETED", "run_id": "EXP-017-PILOT-001", "start_utc": "2026-11-27T08:20:24.304791+00:00", "code_sha": "c718afecce6c65bcccbe8ba2b3e23ba07b2a7294", "gpu_uuid": "GPU-c0ba2c71-aee0-5675-4047-210764508ba5", "completed": {"U0": 20, "U1": 20, "U2": 20}, "current_group": "U2", "current_epoch": 20, "last_utc": "2026-11-27T09:03:11.948685+00:00", "wall_seconds_including_preflight": 2764.9238743782043, "cpu_seconds_including_preflight": 4221.554504000001, "end_utc": "2026-11-27T09:03:14.742268+00:00"}。墙钟包含前置检查与等待，CPU 为进程及可见子进程计数估计；所有检查点和哈希见 checkpoint_manifest.json。

原作者 SOP 将各批次计数平均后按配置 batch 归一；样本加权 SOP 使用总计数除以实际 4500 样本。尾批保留；SOP 为理论计数，非实测能耗。原始逐层阈值、密度、BN、有效扇出与逐样本数据均保存在本目录。

限制：单种子、同一验证集工作点筛选；20轮结果不能代表300轮稳定性。原始 AMP 跳步后手动惩罚语义保留。按轮采样与worker种子固定，检查点恢复了模型/优化器/调度器/scaler/RNG；跨环境、工作进程中断后的逐位可复现性未确认。CPU 统计可能遗漏进程启动及极短暂子进程。

## 独立核查与时间估算

60轮验证逐样本及批次记录已独立重聚合；全部20轮的三组采样顺序及丢弃样本一致。三个最终检查点均以CPU重新装载，模型、优化器动量、调度器及scaler逐项一致，RNG组件和下一轮采样信息存在；未进行额外训练验证跨中断逐位轨迹。详见 checkpoint_recovery_audit.json 与 validation_reaggregation_audit.json。

|组|SOP相对U0节省|准确率相对U0|平均训练时间/轮|含验证和保存的300轮工程估算|
|---|---:|---:|---:|---:|
|U0|0.000%|+0.000个百分点|28.85秒|2.87小时|
|U1|15.984%|-1.067个百分点|40.87秒|3.87小时|
|U2|42.333%|-1.578个百分点|40.83秒|3.86小时|

上述估算仅按当前20轮平均线性外推，未包含启动、额外诊断、哈希审计和共享资源变化。当前是同一训练种子下的验证集工作点筛选，不是论文复现或独立最终测试。
