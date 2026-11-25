# Controlled baseline

先执行 split.py 固定索引，再运行 check.py 和获得授权的 benchmark.py。训练入口只构造 CIFAR-10 训练来源子集；`final_test.py` 是单独的、对预锁定 checkpoint SHA256 校验且拒绝覆盖结果的测试入口。最终训练需单独授权。

候选命令（完整 200 epoch，本轮禁止执行）：

```
CUDA_VISIBLE_DEVICES=<approved-gpu> python experiments/EXP-004/controlled/train.py --config experiments/EXP-004/config.yaml --output experiments/EXP-004/outputs/formal-RUN-001 --confirm-formal-training
```

使用原始 easy_load_data 的增强、顺序采样、SGD / timm 调度器、time-mean logits 与阈值更新；仅训练/验证划分及评估状态隔离是明确偏离。AMP overflow 时仍按 baseline 手动更新阈值，需记录并请理论研究员确认。模型状态文件会保留本地，不纳入 Git。
