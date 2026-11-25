# EXP-004 Baseline 复现准备 — PLANNED

目标：CIFAR-10 ResNet-20、原始 AT-LIF、T=8、eta=0.002、eta2=0；训练种子 42，batch 128，梯度累积 1，SGD lr=0.1，cosine，200 epochs，其他原配置保留。`eta` 传入 ATLIF sp，`eta2` 乘 `regularize_spike`；batch 求和、时间均值及手动更新行为见 EXP-000 报告。配置是准备稿，不是已运行实验。当前脚本 commands.sh 只报阻塞并退出，不能启动训练。

数据目录 `/home/hym/dataset/cifar10/` 下存在 CIFAR-10。建议固定分割种子 2026，将官方训练集独立划出 5000 张验证（剩余 45000 训练），索引与校验值必须在正式训练前保存；最终测试集 10000 张仅在最终检查点确定后评估一次。此为待理论确认的划分方案，训练样本比原始配置减少 10%，直接比较论文结果有局限；实际分割尚未产生。

原入口 `main.py:633-638,766-811` 每轮评估 test 且用 test_acc1 选 checkpoint，违反保留测试集要求，必须先实现并测试独立验证路径。eval 时 update_value 积累，可能影响下一次训练手动阈值更新，如何隔离应先确认；不可静默修改原训练规则。`easy_load_data` 使用 RandomCrop、HorizontalFlip、Cutout，训练 loader 未显式 shuffle，需与论文实际设置核对；不自行变更。原代码构造 checkpoint 的 max_acc1 在比较前存储，恢复语义需验证。

项目目录未发现 .pt/.pth/.ckpt/.safetensors 模型文件；上游 GitHub README 未列预训练下载，仓库页面无 release，可见资源未找到可靠权重，不排除作者其他渠道有权重；未下载模型。无权重校验值/训练来源可报告。

单卡候选 RTX 4090 24GiB；可见 8 张不是已批准训练资源。2026-11-25 检查 GPU 空闲显存约 1.5–24.2 GiB，随共享负载变化；正式 batch=128 是否适合单卡未验证。未执行任何训练吞吐或显存基准，所以不能可靠估计训练总时长或峰值显存；需要经批准后做受控训练步基准，再以 200×ceil(45000/128) 步与实际吞吐估算（不含验证与 I/O）。训练时长和 GPU 数量须审批，当前轮的 1 小时只覆盖本轮冒烟，不授权完整训练。

后续候选命令（当前不可安全运行）：进入 baseline 目录后 `python main.py --config=../experiments/EXP-004/config.yaml`。该命令仍使用官方测试集做选择，必须先完成适配、测试及预算批准。论文 91.06%、FR 0.043、SOP 48.85M 仅作文献参考，并非本项目结果或通过门槛。
