# ROUND-002 科研状态

DEC-002 REPAIR；Q-001 主线、Q-002 竞争解释；理论 v0.1.0。H-001/H-002 UNTESTED；H-003 INSUFFICIENT_EVIDENCE。

- EXP-000 COMPLETED；ROUND-001 结果 `77cb77619ff011f8f78fb3e0c7ef1a0a4cb31c29`。
- EXP-001 COMPLETED（合成机制及单元测试范围）：RUN-004 495 组捕获对照通过；独立模拟及实际 ATLIF R1/R2 通过。实际重置状态由 m 和 s 重建，不是独立观测。
- 随机初始化网络工程冒烟完成（非 EXP-002，不能推断任务性能），19 个 ATLIF 执行一次；发现 eval 非零 sp 累积 update_value。
- EXP-002/003 保持 BLOCKED：无可追溯权重及完整真实网络干预验证；EXP-004 PLANNED（存在阻塞）：独立验证入口与 eval 状态策略待解决，正式训练资源未获批。
- 本轮代码 `b61a7609e63a0c3e6cfec773d3775ac5a29aa66a`；CPU-only，RUN-004 进程 CPU 27.48 秒，冒烟进程 CPU 1.52 秒；未测其他核查工具总 CPU，未训练、未使用 GPU。
- 下一步：确认数据拆分及验证状态策略，开发并测试独立验证入口，测算资源并申请正式训练预算。
