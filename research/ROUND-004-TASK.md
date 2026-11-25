# Codex 科研执行任务：ROUND-004

# AT-LIF Controlled Baseline 正式训练与科研检查点交付

## 一、任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-004
- 科研决策：DEC-004，CONTINUE
- 理论版本：v0.1.1
- 主线科学问题：Q-001，动力学输出补偿充分性
- 竞争性解释：Q-002，脉冲时序与任务敏感性
- 科研假设：H-001、H-002，均保持 UNTESTED
- GitHub 仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-003-controlled-baseline
- 本轮建议分支：research/round-004-baseline-training
- 原始 Baseline SHA：52878cff06f6378fa155c032b7acd23883a72e4a
- 上轮 GPU 基准代码 SHA：4d8272eee513ae5e303d9a8cb3cc97d4aec82b68
- 上轮最终入口代码 SHA：7afbeea1afbe80fccff16636d13457e88153102e
- 上轮结果 SHA：58b3a5324d1fb82992d0c06cf0047029283120cb
- 本轮正式实验：EXP-004-RUN-001（若已占用，分配新的唯一 Run ID）
- 依赖实验：EXP-000、EXP-001、ROUND-003 CPU状态隔离测试及GPU基准
- 优先级：P0 最终前置检查；P1 正式训练

首先读取 AGENTS.md、research/RESEARCH_STATE.md、research/THEORY.md、research/HYPOTHESES.md、research/ROUND-003-TASK.md 和 experiments/EXP-004/REPORT.md。

本轮不要重复 EXP-001 的完整合成测试，也不要重新设计数据划分或实现新的剪枝方法。

若用户已明确批准本轮正式训练资源，所有前置条件通过后直接开展正式训练，不需要再次逐项请求确认。

若尚未得到正式资源批准，只完成 CPU 可执行任务，GPU及完整训练保持未运行。

## 二、ROUND-003 正式审查结论

理论研究员已经实际读取 GitHub 远程提交、数据划分索引、代码、CPU状态隔离日志、GPU原始测速数据及训练入口。

已确认：

1. 数据划分为45000训练、4500验证、500诊断池，三者互斥并完整覆盖50000官方训练样本。
2. 固定256个诊断样本属于诊断池，无重复。
3. CPU A/B/C对照的阈值差异分别为0、0.000124812126、0。
4. 隔离验证保留进入时非零 update_value，A/C网络权重、BN和优化器状态差异均为0。
5. R13 在指定工程对照条件下获得实验支持。
6. CPU小型网络检查点恢复测试通过。
7. GPU基准使用RTX 4090、batch=128、T=8、AMP=True。
8. 训练平均Batch耗时0.198052636秒，含加载吞吐645.217799样本/秒。
9. 峰值allocated显存4792188416字节，约4.46GiB。
10. 峰值reserved显存4942987264字节，约4.60GiB。
11. GPU基准未检测到AMP跳步。
12. 完整200轮训练粗略估计约4.28小时，不包含全部额外开销。
13. 原始Baseline代码未修改。
14. 正式200轮训练未执行。
15. H-001、H-002尚无真实训练网络证据。

本轮主要目标是取得可靠的正式训练检查点，而非继续扩大准备阶段。

## 三、需要解决的剩余问题

ISSUE-008：AMP跳步后的手动阈值更新行为。

ISSUE-009：修复后的单卡GPU路径尚未重新验证。

ISSUE-010：正式训练入口缺少活动剪枝科研指标及完整运行记录。

这些任务只允许进行必要的最小范围实现，不得开展无关代码重构。

## 四、AMP理论决定

原始Baseline训练执行：

scaler.scale(loss).backward()

scaler.step(optimizer)

threshold_update(model, lr)

scaler.update()

当AMP检测到非有限梯度时，optimizer.step可能被跳过，但手动阈值更新仍可能执行。

定义：

I_k = 1：优化器实际更新。

I_k = 0：优化器因AMP溢出跳步。

原始行为可抽象为：

W_next = W + I_k * Delta_W

theta_next = theta + I_k * Delta_theta_optimizer + lr * v

其中 v 为前向活动统计产生的手动阈值更新信号。

本轮理论决定：保留原始AMP跳步时的手动阈值更新行为。

不得擅自增加 if optimizer_stepped 条件来改变原始训练机制。

不得将该问题发展为本轮新算法。

但需完成一次受控AMP测试：

- 使用独立临时模型和优化器。
- 构造正常训练步及人为注入非有限梯度的训练步。
- 确认优化器跳步时实际权重和优化器状态。
- 记录手动阈值更新前的累积量 v、学习率及阈值。
- 验证阈值变化是否与 lr*v 在数值容差内一致。
- 记录 scaler 的变化。
- 确认前向活动统计为有限值。
- 测试结束销毁临时模型，不作为正式训练初始化。

如果出现非预期行为，记录并停止受影响任务，不能擅自修改Baseline核心机制。

## 五、固定数据划分

直接复用：

experiments/EXP-004/splits/indices.json

上轮记录的文件 SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

训练前核查：

- 索引文件校验值。
- 本地数据文件校验值。
- 45000训练样本。
- 4500验证样本。
- 500诊断池样本。
- 256固定诊断样本。
- 所有子集互斥及预处理正确。

不得重新生成已有数据划分。

训练继续采用已确认的原始数据增强和顺序采样方式。

官方测试集不得参与本轮模型选择、超参数选择、训练监控或科研假设验证。

本轮建议保留官方测试集不运行，待后续正式论文实验协议锁定后统一评估。

## 六、正式训练配置

使用 experiments/EXP-004/config.yaml 作为现有配置基础。

目标配置：

- dataset=CIFAR10
- model=ResNet20
- neuron=原始 ATLIF
- T=8
- tau=0.9
- eta=0.002
- eta2=0
- seed=42
- split_seed=2026
- batch_size=128
- num_workers=4
- accumulation_steps=1
- epochs=200
- optimizer=SGD
- momentum=0.9
- nesterov=True
- lr=0.1
- lr_scheduler=cosine
- min_lr=1e-6
- warmup_epochs=0
- warmup_lr=0
- cooldown_epochs=沿用原始实际生效值10
- weight_decay=5e-4，保留原参数组处理
- amp=True
- TET=False
- label_smoothing=0

保存实际解析后的完整配置，包括调度器实际参数及优化器参数组。

若配置未真正驱动程序而由固定代码参数实现，应确认有效值完全一致并保存实际生效配置。

不得擅自修改学习率、Batch Size、T、eta、数据增强、采样器或训练轮数。

任务损失：

logits 的逻辑形状为 [T,B,C]。

z[b,c] = mean_t(logits[t,b,c])

loss[b] = CrossEntropy(z[b,:], label[b])

保持原始 TET=False 行为。

## 七、最后一次 GPU 前置检查

仅在用户明确授权后进行。

使用一张经过授权且实际空闲的 RTX 4090。

建议至少8GiB空闲显存，并核查共享服务器当前负载；不得干扰其他用户任务。

通过 CUDA_VISIBLE_DEVICES 仅暴露所选物理 GPU。

进程内部验证：

torch.cuda.device_count() == 1

逻辑设备为 cuda:0。

不得在导入原始代码后重设 GPU 可见范围。

不得初始化、查询RNG状态或执行计算于未授权GPU。

使用最新训练入口代码，在不超过10分钟的GPU预算内完成：

1. 10个训练Batch预热。
2. 50个训练Batch测速，或满足稳定测量的更少Batch。
3. 少量验证Batch。
4. 原始AMP跳步行为的受控测试。
5. 最新GPU路径的状态隔离A/C对照。
6. ResNet-20 AMP检查点保存与恢复的短步骤测试。
7. 固定数据划分及正式入口运行检查。

保存新的 Run ID，不覆盖原GPU基准记录。

A/C必须使用相同初始模型、优化器、scaler、输入和随机状态。

记录模型参数、阈值、BN、优化器状态及实际数值差异。

在确定性条件下，状态隔离默认采用1e-5绝对误差容差；无法达到时记录真实差异和原因，不事后放宽标准。

验证恢复后下一步训练与不中断训练的一致性，包括 AMP scaler 状态。

前置检查模型及优化器不得用于正式训练初始化。

若GPU路径、AMP、隔离或恢复测试出现关键错误，停止正式训练。

无需为非关键格式问题无限重复测试。

## 八、正式训练指标记录

当前 controlled/train.py 尚未充分记录活动剪枝科研指标。

正式训练前最小化完善日志功能。

每个Epoch至少保存：

- epoch
- learning_rate
- train_loss
- train_accuracy
- validation_loss
- validation_accuracy
- best_validation_accuracy
- training_sample_count
- validation_sample_count
- amp_skipped_steps
- optimizer_steps
- epoch_wall_seconds
- peak_allocated_bytes
- peak_reserved_bytes
- checkpoint_path（适用时）
- checkpoint_sha256（适用时）

逐层额外记录：

- module_name
- epoch
- threshold
- training_firing_rate
- validation_firing_rate
- neuron_state_count
- spike_count

优先使用已有前向输出和无副作用统计。

不允许为统计指标额外执行会改变BN、随机状态或 update_value 的训练前向。

不保留不必要的梯度计算图。

训练和验证准确率、损失按实际样本数加权。

定义真正的全神经元—时间步加权发放率：

FR_global = total_binary_spikes / total_neuron_time_states

同时尽可能保留原作者的 FR 统计口径，并单独命名，例如 FR_author。

不要将逐层均值、全神经元加权平均或缩放输出平均混为一谈。

记录各指标的具体定义。

若某类指标无法可靠获取，标记NA并报告原因，不编造结果。

验证必须保持状态隔离，收集统计量之后恢复原模型相关状态。

对于评估期间的参数、buffers、ATLIF统计和随机状态，保留已通过CPU测试的隔离语义。

正式训练结束后，对锁定的最佳验证检查点，在验证数据上计算：

- Validation Accuracy
- Validation Loss
- 各层及整体 FR
- 参数量
- SOPs

SOPs需明确原作者统计定义、实际计数对象以及是否包含第一层模拟输入或其他非脉冲路径。

不将理论运算量描述为硬件实测能耗。

## 九、检查点与运行管理

正式训练使用新的唯一 Run ID。

不得覆盖历史实验结果。

检查点至少保存：

- model state_dict
- optimizer state_dict
- scheduler state_dict
- AMP scaler state_dict
- epoch
- best_validation_accuracy
- 随机数状态
- 必要的ATLIF内部状态
- 训练配置和数据划分版本

训练选择规则：

以验证集Top-1 Accuracy最高的Epoch作为最佳检查点。

完全并列时保留较早Epoch。

保存最新检查点及最佳验证检查点。

checkpoint_latest 和 checkpoint_best 可以按同一Run的既定规则更新，但须保留完整指标历史和对应Epoch记录。

检查点采用可靠的临时文件写入及原子替换等机制，避免中断导致唯一恢复文件损坏。

若需恢复训练：

- 保留原运行元数据。
- 分配独立恢复段或新Run ID并记录父Run。
- 不重新覆盖原始 run_metadata.json。
- 校验代码、配置、划分文件和检查点。
- 从正确Epoch继续。
- 保持原最佳验证准确率。
- 不将续训当作独立随机种子实验。

不要求保存全部200个Epoch的完整模型权重。

大型检查点保留于服务器本地或用户指定外部存储。

GitHub必须记录其实际位置、SHA256、大小、对应Epoch、模型配置和训练Run ID。

不要将大型权重直接提交Git。

## 十、正式训练资源及终止条件

只有用户另外批准正式训练资源后才能开始。

建议的待批准预算：

- 一张RTX 4090。
- 总GPU占用上限6小时，包含本轮GPU前置检查。
- 前置检查不超过10分钟。
- 一次200Epoch正式训练。
- 单一随机种子42。
- 不进行多GPU训练。
- 不进行超参数搜索。

未获得批准时，不得自行将上一轮测速授权扩展为正式训练授权。

正式运行前记录资源预算、实际开始时间和最大停止时间。

设置可执行的预算控制机制，不仅在文档中声明预算。

出现以下情况应停止相关训练：

1. 达到批准的最大运行时间。
2. CUDA OOM。
3. 非有限训练损失。
4. 非有限或非正阈值。
5. 检查点写入失败，无法保证恢复。
6. 训练数据划分或代码版本不一致。
7. 无法保证GPU隔离。
8. 出现持续严重的AMP数值异常。

预先记录AMP异常处理规则：

- 单次AMP梯度溢出不自动判定实验失败，保留原始更新行为并记录。
- 连续10次优化器跳步，或者任意完整Epoch中跳步比例超过5%，停止训练并报告。
- 这些阈值是本项目的工程安全停止条件，不是科学性能判定标准。

如接近预算上限，优先在安全的Epoch边界保存完整检查点。

不要为了完成200轮而超出已批准预算。

如在第200轮前停止，将实验标记为PARTIAL或FAILED等实际适用状态，并保存可恢复检查点。

不得偷偷增加GPU数量或修改训练超参数。

## 十一、正式训练执行顺序

1. 检查仓库和当前Git状态。
2. 从ROUND-003最终结果版本创建独立实验分支。
3. 核查长期科研规范及固定数据划分。
4. 最小化修复训练指标与运行记录。
5. 核查配置、优化器和调度器实际行为。
6. 保存并提交本轮预注册任务及执行代码。
7. 核查用户正式GPU授权。
8. 完成最终单卡GPU前置检查。
9. 检查全部正式训练进入条件。
10. 使用独立新模型，重新设置训练seed=42。
11. 启动200Epoch正式训练。
12. 每轮保存真实指标和检查点。
13. 选择并锁定最佳验证检查点。
14. 在验证数据上计算FR和SOPs等指标。
15. 保存报告、曲线及原始日志。
16. 更新科研状态并提交GitHub。

所有前置检查通过且已有用户资源授权时，直接开展正式训练，不再返回只有计划的回复。

本轮不运行 EXP-002 或 EXP-003 正式科学假设诊断，不实现新方法。

## 十二、结果评价与交付

原论文报告值：

CIFAR-10，ResNet-20，T=8，lambda_theta=0.002，lambda_s=0。

Accuracy=91.06%。

Avg.FR=0.043。

SOPs=48.85M。

以上仅为原论文参考值。

Controlled Baseline使用45000训练样本和独立验证，且隔离评估状态，不属于完全相同协议下的复现。

不能将其验证准确率与论文测试准确率直接作为公平性能比较。

不设置必须达到论文准确率的硬性目标。

训练成功结束不代表H-001、H-002成立。

生成至少以下图表：

1. Train/Validation Loss曲线。
2. Train/Validation Accuracy曲线。
3. 全网络FR曲线。
4. 各层FR随Epoch变化图。
5. 各层Threshold随Epoch变化图。
6. 学习率曲线。
7. AMP跳步统计。
8. Accuracy与FR的训练演化关系。

所有图表保留原始数据。

不得隐藏训练异常、准确率下降或负面结果。

## 十三、科研状态更新

新增 research/ROUND-004-TASK.md。

追加DEC-004至DECISION_LOG.md：

- ROUND-003验收基本通过。
- R13在CPU工程对照中获得实验支持。
- ISSUE-006独立验证协议已实现。
- ISSUE-007状态隔离CPU测试通过。
- ISSUE-008保留原始AMP更新行为并监控。
- ISSUE-009完成最终单卡GPU前置检查。
- ISSUE-010补充正式训练科研指标。
- 主线Q-001不变。
- 理论版本v0.1.1不变。
- 本轮正式训练EXP-004。

更新：

- RESEARCH_STATE.md
- EXPERIMENT_PLAN.md
- EXPERIMENT_REGISTRY.csv
- BASELINE.md
- DECISION_LOG.md
- EXP-004/REPORT.md
- EXP-004/status.json

补记ROUND-003结果SHA：

58b3a5324d1fb82992d0c06cf0047029283120cb

实验状态按实际执行情况填写。

H-001和H-002均保持UNTESTED。

EXP-002和EXP-003在理论研究员审查正式训练权重前保持未执行。

## 十四、GitHub交付要求

保存并提交：

- 正式任务协议。
- 最终执行代码。
- 实际生效配置。
- 运行命令。
- GPU前置检查报告及原始指标。
- 正式训练逐Epoch原始数据。
- 逐层发放率及阈值数据。
- 实际运行日志。
- 全部失败和异常记录。
- 模型检查点清单及校验信息。
- 实验图表。
- 最终实验报告。
- 科研状态文件。

大型检查点不得直接提交GitHub。

不提交私钥、凭证、数据集或无关用户文件。

不合并main，不force push。

提交时记录实际运行代码SHA和结果提交SHA。

远程同步失败时报告真实原因。

## 十五、最终返回信息

- ROUND-004 GitHub分支链接。
- 实际运行代码Commit SHA。
- 结果提交Commit SHA。
- GPU最终前置检查结果。
- AMP受控测试结果。
- 正式训练Run ID。
- 正式训练执行状态。
- 实际完成Epoch数量。
- 最佳验证Epoch。
- 最佳验证Accuracy和Loss。
- 最佳检查点FR及SOPs。
- 最佳检查点实际存储位置及SHA256。
- 最新恢复检查点位置。
- 逐Epoch指标及逐层指标路径。
- 实际GPU资源消耗。
- 全部运行异常。
- 是否存在不满足科学诊断前置条件的问题。

本轮唯一核心目标是取得可靠、可追溯的AT-LIF正式训练检查点，为后续真实网络动力学假设验证提供依据。

请实际执行获得授权的任务，不要仅生成训练计划。禁止编造实验结果。