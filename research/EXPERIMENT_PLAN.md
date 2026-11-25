# ROUND-001 预注册方案

完整协议、指标、阈值及预算见 [ROUND-001-TASK.md](ROUND-001-TASK.md)，原样保存于运行前。EXP-000 审查；EXP-001 CPU 合成机制及实际 ATLIF 单元测试；EXP-002 仅在可靠权重、数据及其他前提齐备后执行；EXP-003 仅在 EXP-002 所有预设门槛通过后执行。

资源估计：CPU 少于 10 分钟，GPU 0；硬预算 CPU 累计 30 分钟、单 GPU 累计 1 小时；不训练、不下载。R1/R2 相对误差容差 float64 1e-10，float32 1e-5。
ROUND-002 修复与冒烟协议见 ROUND-002-TASK.md；EXP-004 尚未正式训练，安全入口、数据划分和资源预算待解决，见 ../experiments/EXP-004/REPORT.md。

ROUND-003 协议见 ROUND-003-TASK.md；用户只授权 EXP-004-BENCH-01 单卡测量，不授权完整训练。已固定分层索引及独立验证入口；完成正式 GPU 路径审查与资源批准后方可启动训练。
