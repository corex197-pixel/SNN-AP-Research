# ROUND-004 正式训练

单进程仅暴露已批准 GPU UUID；先保存协议和代码 SHA，运行 PREFLIGHT-001 全部通过，检查固定数据划分，才能以 seed=42 创建新的正式模型。正式训练命令（仅在预检通过且预算仍有效时）：

```
CUDA_VISIBLE_DEVICES=<approved-gpu-uuid> python -u experiments/EXP-004/round004/train.py --run-id EXP-004-RUN-001 --deadline-utc <absolute-utc-deadline> --approved-gpu-uuid <approved-gpu-uuid> --code-sha <committed-code-sha>
```

最大停止时间是前置检查开始后 8 小时；训练入口提前 120 秒安全停止，逐 epoch 原子保存 latest/best 和对应 SHA。每 epoch CSV 保留完整历史，AMP 溢出保留 baseline 更新行为；连续10次跳步或整轮超过5% 停止。运行失败时保留已产生检查点和日志。正式恢复入口尚未实现，若中断只交付 checkpoint 并记录为 PARTIAL，不擅自重启。

FR_global 按所有 ATLIF 实际输出的二值脉冲数量/神经元时间状态数量统计；FR_author 为各模块的 batch 等权 FR 均值再按层等权，仅为辅助口径，不假称等同于加权 FR。验证在 `isolated_eval` 内进行并在退出前收集无梯度数据；SOP 定义见最后汇总输出。正式测试集不用于本轮。
