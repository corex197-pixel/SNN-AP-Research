# Baseline

本项目初始导入 SHA `52878cff06f6378fa155c032b7acd23883a72e4a`；此 SHA 是本项目导入版本，不是上游作者提交 SHA。原仓库元数据缺失，上游具体版本尚未确认。工程审查见 experiments/EXP-000/REPORT.md。

ROUND-002：项目目录及作者仓库可见资源未找到可靠预训练权重；上游无列出的 release，不能排除其他渠道。独立验证拆分与 eval 累积 update_value 问题见 ../experiments/EXP-004/REPORT.md。

ROUND-003：Controlled Baseline 明确使用 45000/4500/500 的训练来源分层划分和验证状态隔离；与论文全量训练集不同。实际吞吐、评估污染和原始 CUDA_VISIBLE_DEVICES=4 副作用见 EXP-004/REPORT.md。
