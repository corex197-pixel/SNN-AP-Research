# ROUND-001 科研状态

- Q-001 主线；Q-002 竞争解释；理论 v0.1.0；DEC-001 CONTINUE。
- EXP-000 COMPLETED：代码及环境审查完成；未发现可溯源可靠权重。
- EXP-001 PARTIAL：合成 R1/R2 和实际 ATLIF 前向 R1/R2 数值通过；独立无捕获对照未完成。详见 ../experiments/EXP-001/REPORT.md。
- EXP-002 BLOCKED：无可靠权重及实际网络捕获核验；EXP-003 BLOCKED：进入门槛无结果。
- H-001/H-002 均 UNTESTED；H-003 INSUFFICIENT_EVIDENCE。无真实模型性能或因果结论。
- baseline 导入 `52878cff06f6378fa155c032b7acd23883a72e4a`，最终有效合成运行代码 `a90a09ba8b1cd34d943774db275c5aefb58f7bb5`。未训练、未下载数据、未使用 GPU；两次有效测试合计墙钟约 43 秒，CPU 时间未单独测量。
- 下一步提供可验证权重、来源与配置；先补无捕获对照及实际网络输入和模块可采集性核验，再按原协议评估 EXP-002 前置条件。
