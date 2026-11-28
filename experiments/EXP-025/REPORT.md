# EXP-025：结构化剪枝 Baseline 静态准入审查
状态 COMPLETED（静态审查已完成；所有候选仍 UNDER_REVIEW、存在阻塞，非 ENGINEERING_ADMITTED）。任务/审查记录版本 `0d8138080f3bf76a6dd642356c534df15a5f7062`，父结果 `7d5bdba3532ccb5b7dc3a8d69053c39314fe4a93`。未进行训练、推理或 GPU 操作。

| 候选 | 加权分 / 10 | 准入结论及主要障碍 |
|---|---:|---|
| C-DPSS | 6.1 | 有条件优先审查；许可证不明、CIFAR 官方测试集被当验证用于最佳模型选择、全文/权重未知 |
| SCA | 5.0 | 公式明确；官方代码、许可证、预训练权重未知 |
| Dynamic Spatio-Temporal | 4.15 | 时序研究有关；固定代码权重级掩码，真实通道删除未确认、许可证未知 |

评分由科学问题/创新/SNN特异性/代码权重/可复现性/资源/公平结构化七维分别以 20/20/10/15/15/10/10% 加权，原始分数和限制见 `baseline_comparison.csv`。其为判断而非训练实测，未知项保守扣分，资源未测量。论文和固定源码链接见 `candidate_sources.json`。

SCA Eq.(4) 是跨样本与时间平均的空间膜电位 L1 范数，全局排序的空间面积尺度效应值得检验，不能声称已验证实现缺陷。C-DPSS 的 `--indicate_method test` 仅保存路径标签，但 CIFAR 数据函数将 test_loader 返回作 valid_loader，而主流程用 valid_loader 选最佳模型，是明确的原始代码隔离风险；详细路径见 `data_protocol_audit.md`。Dynamic 已包含基于 KL 的时间步选择，时序信息本身存在重复风险。建议进一步考察 Q-013A 任务贡献与膜电位幅度失配和 Q-013B 跨空间分辨率得分尺度；属于假设，不宣称已发现新算法。

EXP-024 完成且原性能推进门槛失败；H-024 SUPPORTED 仅限指定线性读出及固定结构，H-025 PARTIALLY_SUPPORTED 仅相对 S3，未通过 S1/S2 门槛。RUN-001 重构失败及严格 FP32 与旧默认后端 45 个预测差异保留；EXP-023 训练准入测试偏差保留。停止 M5 正式推进，保留 UniSparse Controlled Baseline。下一阶段只提出条件性计划，需理论研究员审查及新授权。详细文件见本目录，资源计量缺口见 limitations.md。
