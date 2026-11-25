# EXP-004 Baseline 复现准备 — PLANNED

目标：CIFAR-10 ResNet-20、原始 AT-LIF、T=8、eta=0.002、eta2=0；训练种子 42，batch 128，梯度累积 1，SGD lr=0.1，cosine，200 epochs，其他原配置保留。`eta` 传入 ATLIF sp，`eta2` 乘 `regularize_spike`；batch 求和、时间均值及手动更新行为见 EXP-000 报告。配置是准备稿，不是已运行实验。当前脚本 commands.sh 只报阻塞并退出，不能启动训练。

数据目录 `/home/hym/dataset/cifar10/` 下存在 CIFAR-10。建议固定分割种子 2026，将官方训练集独立划出 5000 张验证（剩余 45000 训练），索引与校验值必须在正式训练前保存；最终测试集 10000 张仅在最终检查点确定后评估一次。此为待理论确认的划分方案，训练样本比原始配置减少 10%，直接比较论文结果有局限；实际分割尚未产生。

原入口 `main.py:633-638,766-811` 每轮评估 test 且用 test_acc1 选 checkpoint，违反保留测试集要求，必须先实现并测试独立验证路径。eval 时 update_value 积累，可能影响下一次训练手动阈值更新，如何隔离应先确认；不可静默修改原训练规则。`easy_load_data` 使用 RandomCrop、HorizontalFlip、Cutout，训练 loader 未显式 shuffle，需与论文实际设置核对；不自行变更。原代码构造 checkpoint 的 max_acc1 在比较前存储，恢复语义需验证。

项目目录未发现 .pt/.pth/.ckpt/.safetensors 模型文件；上游 GitHub README 未列预训练下载，仓库页面无 release，可见资源未找到可靠权重，不排除作者其他渠道有权重；未下载模型。无权重校验值/训练来源可报告。

单卡候选 RTX 4090 24GiB；可见 8 张不是已批准训练资源。2026-11-25 检查 GPU 空闲显存约 1.5–24.2 GiB，随共享负载变化；正式 batch=128 是否适合单卡未验证。未执行任何训练吞吐或显存基准，所以不能可靠估计训练总时长或峰值显存；需要经批准后做受控训练步基准，再以 200×ceil(45000/128) 步与实际吞吐估算（不含验证与 I/O）。训练时长和 GPU 数量须审批，当前轮的 1 小时只覆盖本轮冒烟，不授权完整训练。

后续候选命令（当前不可安全运行）：进入 baseline 目录后 `python main.py --config=../experiments/EXP-004/config.yaml`。该命令仍使用官方测试集做选择，必须先完成适配、测试及预算批准。论文 91.06%、FR 0.043、SOP 48.85M 仅作文献参考，并非本项目结果或通过门槛。


## ROUND-003 Controlled Baseline — IMPLEMENTED_NOT_RUN

完整 200 epoch 正式训练未执行、未授权。实现入口 `controlled/train.py` 与独立 `controlled/final_test.py`；源 baseline 核心代码未修改。前者仅加载 CIFAR 官方训练来源的训练/验证子集；以验证 accuracy 严格大于才更新最佳（平局保留早期），样本加权损失与准确率；后者检查预锁定 checkpoint SHA，并对单个输出路径拒绝覆盖。该路径级约束不是全局测试集使用次数治理，须以科研流程控制。

### 数据划分

`split.py` 以 torchvision 完整性检查通过的本地 CIFAR-10 train 为来源，RandomState(2026) 按类洗牌：45000/4500/500，每类分别 4500/450/50；互斥并集 50000。诊断池使用独立 RandomState(2026) 无放回固定抽取 256。索引和批文件 SHA256 存于 `splits/indices.json`，此文件 SHA256 `69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165`。验证和诊断使用确定性变换；训练沿用 RandomCrop、HorizontalFlip、Cutout 和原始 sequential sampler（未 shuffle）。CIFAR 完整性检查可能读取测试文件进行哈希，但未加载测试样本作为训练或选模。训练集比作者完整训练集少 10%，不可声称原论文同条件复现。

### A/B/C 与检查点

`runs/EXP-004-CHECK-002/check.log`：eval 前累计值 0.123；原始未隔离验证后 0.1242488673；隔离后 0.123。执行同一步训练，D_theta_B=0.000124812126，D_theta_C=0；权重、BN 和优化器 C/A 最大差均 0；B/C 评估输出最大差 0；异常退出状态及 RNG 恢复通过。R13 的 lr*V_val 预测在浮点精度内一致；该示例非真实正式训练污染量。使用小型 CPU 网络，不能外推到整个训练。

`runs/EXP-004-RESUME-001/run.log`：CPU 小型网络的检查点恢复后下一训练步，模型参数与优化器动量最大差均 0，恢复最佳验证准确率 0.73、epoch 0；观察到权重和阈值更新且阈值有限正值。未完成正式 ResNet-20 AMP 情况的跨中断训练一致性测试；不能把 CPU 小模型测试当完整训练验收。`eval` 隔离保留进入时非零 update_value、模式、BN buffers、阈值、ATLIF 统计、memory 状态及当前设备 RNG；其他未知模块自定义可变属性无法保证覆盖。

### GPU 资源测量

用户批准单卡测量；RUN-001 因 baseline main.py 导入强制 `CUDA_VISIBLE_DEVICES=4`，所选 ordinal 6 无效，未执行训练；保留失败记录。移除副作用后 RUN-002 使用物理 GPU 6 RTX 4090（开始空闲 23.25 GiB），10 批预热、50 批测量、3 批验证，不保存模型。测量前 6 号利用率为 0%，设备使用情况原文在 metrics.json。同步计时，batch=128、T=8、AMP GradScaler 启用；期间检测到 0 次 scaler 降低（0 不证明溢出行为已完整验证）。训练时间均值 0.1981s、中位数 0.1972s、P90 0.2021s；数据加载等待均值 0.000330s；含加载平均吞吐 645.2 样本/s；3 验证 batch 0.607s。峰值 allocated 4.46 GiB、reserved 4.60 GiB，含隔离状态快照。完整 200 epoch 线性估计 4.28 小时（352 训练和 36 验证批次/epoch），不含完整日志、checkpoint、竞争负载、测试及波动，非预算保证。

基准运行时旧版本 `manual_seed_all()` 及 `get_rng_state_all()` 可能初始化或触及其他可见 GPU 的 RNG 状态；实际训练计算只安排于 GPU 6。已修复为仅操作当前设备 RNG；该修复后的 GPU 路径尚未重新测量。此行为与单卡隔离预期不符，如实保留。核心 AMP 跳步后原代码仍无条件执行手动阈值更新；本适配保留该行为，尚未人为触发 overflow 验证，需理论研究员评估。

### 未完成及批准

EXP-004 仍 IMPLEMENTED_NOT_RUN，EXP-002/003 BLOCKED，H-001/H-002 UNTESTED。正式训练前应审阅 AMP overflow、最终代码单卡 GPU 状态隔离及 AMP checkpoint 恢复，用户须单独批准约 4.28 小时估计之外的实际单卡预算和最大停止条件。配置为 `config.yaml`；候选命令见 `controlled/README.md`，未运行。随机临时模型的损失和短基准不是科研性能结果。


## ROUND-004 — 前置检查未全部通过，正式训练未启动

用户批准单卡含预检上限 8 小时和一次 200 epoch 正式训练；批准不涵盖占用其他人正在运行的 GPU。预注册代码提供逐 epoch 损失、准确率、分层 FR、阈值、AMP 跳步、原子检查点及预算控制；未执行正式训练，尚无训练结果、检查点、FR、SOP 或训练曲线。

PREFLIGHT-001 在 1 个可见 RTX 4090、固定划分校验通过后，10 批预热与 10 批测速、验证冒烟、GPU A/C 和 AMP 检查点恢复通过：模型、优化器、scaler 最大差均为 0。AMP 注入 inf 确认优化器跳步和 scaler 减半，但活动量 v=0，手动阈值更新为零，仅验证了零活动情况，不能证明非零活动更新符合协议。

PREFLIGHT-002 加入要求 v>0 的检查，但执行开始时原空闲设备被其他进程占用，GPU 可用显存 8285323264 字节，小于 8 GiB 安全阈值；按协议中止，未进入 AMP 测试。随后所有设备均有其他计算负载，无法满足“实际空闲、不干扰其他任务”。前两次检查原始记录在 round004/PREFLIGHT-001 与 PREFLIGHT-002；预算开始和绝对截止见 round004/budget.json。

后续只有在实际空闲单卡、原授权及预算仍有效时，才可对非零 v 的 AMP、最终 GPU 前置条件完整复验；不能因零值检验或过期预算启动正式训练。如后续恢复，必须扣除已用 GPU 时间，并确保仍遵守用户批准的总上限及原协议。
