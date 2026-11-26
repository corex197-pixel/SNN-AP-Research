# Codex 科研执行任务：ROUND-010

# AT-LIF 同计算预算性能评价与归一化感知活动剪枝

## 一、任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-010
- 科研阶段：阶段3—4，候选方法优化及性能先导验证
- 科研决策：DEC-010，REFINE + IMPLEMENT
- 理论版本：v0.1.6
- 科学问题：Q-007，BN统计量、阈值及活动预算之间的耦合
- 科研假设：H-009、H-010
- 实验编号：EXP-009、EXP-010
- 候选方法工作名称：NAP-AP
- GitHub仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-009-budget-distillation
- 本轮建议分支：research/round-010-normalization-calibration
- 上轮实际执行SHA：93ad0a558570458b2ca9066ac127a13acd5a435c
- 上轮结果SHA：1cf8ac6505e590b05d4c41a775054409700d11ab
- 依赖实验：EXP-004、EXP-007、EXP-008
- 优先级：P0同预算评价；P1 BN机制诊断和校准；P2补充30%预算结果

首先读取AGENTS.md、THEORY.md、HYPOTHESES.md、RESEARCH_STATE.md、ROUND-009-TASK.md和EXP-008全部必要结果记录。

本轮禁止梯度训练、继续微调、修改神经元动力学及使用官方测试集。

## 二、ROUND-009正式科研结论

EXP-008有效完成。

原始真实验证结果：

B0：

- Accuracy=0.9171111111111111
- Loss=0.28356405380945504
- SOPs=55023962.13333333

P0：

- Accuracy=0.908
- Loss=0.2760692870493078
- SOPs=43652928.192

F0/F1：

- Accuracy=0.898
- Loss=0.31414590278950116
- SOPs=36196294.542222224

F2/M2：

- Accuracy=0.9022222222222223
- Loss=0.33312663960213024
- SOPs=38083756.75022222

F1和M2的全部训练批次预算违反比例为0。

F0/F1的模型状态逐张量相同，F2/M2的模型状态逐张量相同。

R28：

当C_proxy/C0<=0.8时，平方hinge SOP惩罚及其对模型参数的梯度均为0。

因此本轮无法识别SOP约束的实际干预作用。

H-007：

CURRENT_EVIDENCE_AGAINST，仅针对当前微调协议下预期活动回升的预测。预算约束干预效果仍无法判断。

H-008：

INSUFFICIENT_EVIDENCE。

M2比F1准确率高0.4222个百分点，但SOP降幅相差3.4341个百分点，未达到同预算比较条件。

M2没有达到原整体先导筛选门槛。

不得事后改变ROUND-009判定标准。

保留真实0.8门槛梯度检查在正式训练启动后完成这一协议偏差。

## 三、本轮研究目标

1. 将原始模型、普通微调模型、自蒸馏模型重新校准至相近实际SOPs预算，公平比较性能。
2. 在冻结模型和相同输入条件下测量BN训练/推理模式对脉冲活动和任务输出的影响。
3. 比较普通Adaptive BN和逐层推理状态BN统计量校准。
4. 判断是否能够在无需梯度训练的情况下进一步改善Accuracy–SOPs权衡。

不要求实验必须支持预期结果。

## 四、科研假设

H-009，UNTESTED：

固定权重、阈值和输入时，将BN由固定运行统计量切换到当前Batch统计量，可能导致具有实际幅度的全网络SOPs变化。

本轮以相对于eval模式至少5%的SOPs绝对相对变化作为探索性实际幅度参考。

H-010，UNTESTED：

在相近实际计算预算下，按照网络执行顺序重新估计推理状态下的BN总体统计量，相比原始BN和普通Adaptive BN可以改善任务性能。

H-009成立不等于BN已经被证明是ROUND-009训练初期性能下降的主要原因。

H-010成立也不等于本方法已经具有独立学术原创性。

## 五、固定检查点

原始Baseline：

experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth

SHA256：

c068f54379cb5d29bcfa5fe2d9de676cd3cc153433ada1ed5aa1c66355d0b32a

F0：

experiments/EXP-008/runs/EXP-008-F0-RUN-001/epoch-005.pth

SHA256：

18116a671abb235fc56f67e9b4d02feb55bcbde7f0e311f55e23b7e7cab42d2a

F2：

experiments/EXP-008/runs/EXP-008-F2-RUN-001/epoch-005.pth

SHA256：

47ec18bee4fc1ac6c50af170abcbac9d3461576d977429d0aacce208845b2078

必须核查检查点实际存在性、哈希、模型配置和参数有限性。

不得根据本轮诊断结果改选第1—4轮检查点。

F1、M2与对应模型状态一致，不再重复评价。

所有原始检查点只读，不覆盖。

## 六、固定数据及评估协议

数据划分：

experiments/EXP-004/splits/indices.json

SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

固定500张训练来源校准集：

experiments/EXP-007/calibration_indices.json

SHA256：

b849b74ef19cb4a09df756c54871122f3b02aa047c704d0436feab15a38bde52

沿用原500张，不重新抽样。

校准及搜索使用确定性评估预处理。

最终评价使用原4500张验证集。

验证集不参与阈值搜索、BN统计量估计或候选配置选择。

正式测试集继续隔离。

原500张科研诊断池不使用。

固定配置：

- CIFAR-10
- ResNet-20
- 原AT-LIF
- T=8
- tau=0.9
- FP32
- batch_size=128
- num_workers=4
- 固定样本顺序
- torch.no_grad()
- 无优化器
- 无梯度更新
- 全部阈值为正
- 手动阈值更新禁用

除专门的BN模式诊断及普通Adaptive BN校准外，全模型保持eval模式。

所有前向结束后正确清理ATLIF内部统计状态。

任务损失继续使用时间平均logits的CrossEntropy。

## 七、SOPs统计

复用原作者SOPMonitor及EXP-008一致的运算统计范围。

排除首层模拟输入卷积。

不增加原口径未统计的线性层或其他运算。

不得将SOPs解释为实测能耗或参数稀疏率。

可复用ROUND-009的可微计数代理进行额外一致性验证，但本轮所有正式评价使用原Monitor。

重新评价B0和P0。

确认与EXP-008结果具有预期一致性；记录差异。

主要相对节省使用本轮相同评估协议的B0作为分母。

校准集基准记为C0_cal。

原值参考54945117.504，实际重新测量后报告差异。

## 八、EXP-009：已有模型同预算评价

定义原始Baseline的19个阈值为theta_ref[l]。

所有候选模型均采用：

theta[l] = alpha * theta_ref[l]

注意：

F0/F2检查点中的阈值已经乘过1.25。

不能直接再用alpha乘当前保存阈值。

必须从原Baseline检查点读取theta_ref并赋值。

实验组：

C0：原始权重和原BN统计量。

C1：F0第5轮权重及对应BN统计量。

C2：F2第5轮权重及对应BN统计量。

固定候选倍率：

[0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.5, 1.75, 2.0]

每次候选均独立恢复原始对应模型状态。

在500张校准集上评价：

- SOPs
- FR_global
- FR_author
- Accuracy
- CrossEntropy

主要目标：相对于C0_cal节省20% SOPs。

辅助目标：节省30% SOPs。

对于每一目标，优先选择节省比例处于[target,target+0.02]的候选。

如存在多个，选择与目标节省比例最接近的候选；并列时选择较低alpha。

不得根据验证性能或校准准确率改变上述预算匹配规则。

如果固定网格没有匹配候选，允许在相邻且跨越目标的候选区间进行最多3次中点细化。

如有多个跨越区间，优先选择端点SOP降幅距离目标之和最小的区间，并列时选择alpha较低的区间。

不假定全网络SOPs关于alpha严格单调。

所有已测候选均保留。

若仍无法匹配预算，报告最近达到的实际结果及匹配失败原因。

必须先锁定候选配置，再执行4500张验证集评价。

若C1/C2在20%近似同预算下均无优于C0的实用性信号，不继续增加同类微调训练。

## 九、EXP-010A：BN模式机制诊断

首先使用原始Baseline、P0和F0。

对于每个模型，从固定校准集获取完全相同的确定性评估预处理输入。

构建两组独立模型副本：

A：所有模块eval，BN使用运行统计量。

B：全模型eval，仅BN模块切换到train模式，使用当前Batch统计量。

无梯度计算，不更新任何可训练参数。

B组产生的BN运行统计量变化仅限临时副本，不能污染A组或后续实验。

记录：

- 全网络SOPs。
- 每层FR。
- 分类损失及准确率。
- 每个BN的running mean和running variance。
- 每个BN实际输入的通道均值和方差。
- 两种模式下实际输出差异。

定义：

D_mode=(C_batchBN-C_evalBN)/C_evalBN

同时报告有符号值和绝对值。

若至少P0的abs(D_mode)>=0.05，且重复同条件计算稳定，则报告达到H-009预设探索性幅度。

使用相同Batch Size和数据顺序。

BN训练模式会使一个样本的输出依赖同Batch其他样本，因此不得将这里的结果描述为正常单样本部署性能。

禁止在验证集或测试集上以BN训练模式更新运行统计量。

若预算允许，可增加固定随机种子的训练增强输入，分别重复两种BN模式，以描述增强与BN的影响；不将该补充结果替代主要无增强对照。

## 十、EXP-010B：普通Adaptive BN对照C3

每个候选配置均从原始Baseline独立复制。

先设置绝对阈值倍率alpha。

冻结权重、BN affine及全部阈值。

模型整体eval，仅BN模块train。

重置BN running mean、running variance、num_batches_tracked。

设置momentum=None，使用累计平均更新方式。

在固定500张校准集上执行完整一次前向遍历。

正确清理ATLIF状态。

记录实际Batch数及最后一个Batch大小。

明确普通BN累计平均的Batch权重及方差估计方式。

校准结束后将全模型恢复eval。

冻结校准后的BN统计量。

使用原Monitor重新测量校准后的SOPs。

再根据固定预算匹配规则选择alpha。

每个候选alpha都必须独立进行BN校准，不能将前一候选的BN统计量继续用于后一候选。

## 十一、EXP-010C：逐层推理状态校准M3

理论依据R29：

对于直接连接BN的ATLIF，在gamma>0时：

a[t] >= mu + sqrt(var+eps)/gamma * (theta-beta-tau*u[t-1])

等价于该时刻的发放条件。

gamma<0时不等式反向。

gamma=0时不能使用上述除法。

残差连接需要考虑shortcut输入。

该式不意味着跨时间膜电位状态固定。

### R31校准公式

对每个BN模块实际输入：

逻辑维度[T,B,C,H,W]，或框架内部等价的[T*B,C,H,W]。

必须先确认真实张量布局和channel轴。

沿除channel轴之外的所有维度统计：

count[c] = 实际元素数

mean[c] = sum(a[c])/count[c]

var[c] = sum((a[c]-mean[c])^2)/count[c]

采用总体方差，即correction=0。

跨Batch使用正确的合并均值与二阶中心矩统计，例如Chan/Welford合并算法。

不要直接平均Batch方差。

统计使用float64或经过严格对照的数值稳定等价实现。

最终转换为原BN buffer的dtype。

不改变BN eps、gamma、beta。

### 校准算法

Input: 原始Baseline、固定alpha、500张校准图片。

1. 独立复制原始模型。
2. 设置19个ATLIF的theta=alpha*theta_ref。
3. 冻结全部参数。
4. 全模型保持eval。
5. 根据实际前向调用顺序确定所有BN模块。
6. 对第一个BN注册输入统计hook。
7. 在整个500张校准集上执行eval前向并累计该BN输入统计。
8. 更新当前BN的running_mean和running_var。
9. 移除当前统计hook。
10. 保留已完成校准的上游BN。
11. 对下一个BN重复步骤6—10。
12. 所有BN完成后执行正式校准集评价。
13. 保存所有BN统计量、阈值及计算成本。

每个BN统计过程中不改变其他BN的运行统计量。

同一模块多次调用、非前馈依赖或者动态执行顺序发生变化时，先报告适用性问题，不得强行套用该算法。

对于当前前馈ResNet20，预期可以使用实际拓扑执行顺序逐层校准。

所有校准模型必须保持原权重、BN affine和神经元机制不变。

只允许BN running_mean、running_var及必要的统计计数buffer变化。

每个候选alpha都独立重新校准BN。

### 候选搜索

与EXP-009使用相同固定倍率网格、20%主要目标、30%补充目标及最多3次中点细化规则。

C3与M3均按校准集真实SOPs进行配置选择。

如果资源不足，优先完成20%主要预算。

任何未完成的组必须记录PARTIAL，不能仅保留最有利的候选。

## 十二、必要正确性检查

必须全部在正式实验启动前完成：

- 原始检查点和数据哈希。
- 实际BN及ATLIF模块顺序。
- BN输入张量布局。
- 人工小张量均值及方差合并测试。
- 不完整Batch统计正确性。
- 普通BN校准与逐层BN校准仅修改允许的buffer。
- 原权重及BN affine不变。
- 阈值绝对倍率正确。
- 全部BN统计量有限、方差非负。
- 数值统计hook不改变模型前向结果。
- 原Monitor重复使用无计数污染。
- ATLIF状态正确隔离。
- 临时候选不会污染后续模型。
- 原始检查点SHA256不变。

本轮不需要重复完整EXP-001或ROUND-009反向梯度测试。

若涉及BN统计量重估的新正确性检查失败，暂停相关正式实验。

## 十三、正式评价及判定标准

全部最终阈值和BN统计量在校准集上锁定后，再使用4500张验证集评价。

评价组：

B0、P0、C0、C1、C2、C3、M3。

如20%和30%目标选中相同配置，可以复用真实评价结果。

每组记录：

- Accuracy
- CrossEntropy
- SOPs
- SOPs相对于B0的降幅
- FR_global
- FR_author
- 每层FR及阈值
- 参数量
- 校准耗时
- 完整网络前向次数
- 处理样本总次数
- GPU峰值显存

保存逐样本损失、预测类别和正确性。

使用2000次配对Bootstrap，seed=2027。

主要比较：

C1 vs C0

C2 vs C0

C2 vs C1

C3 vs C0

M3 vs C0

M3 vs C3

预算匹配条件：

相对B0的实际验证SOP降幅差不超过2个百分点。

若预算不匹配，不宣称公平同预算优势。

保留实际Pareto支配关系的描述。

H-010主要筛选门槛：

1. M3实际验证SOPs降幅至少20%。
2. M3准确率相比B0下降不超过1个百分点。
3. M3与C0、C3均满足预算匹配。
4. M3准确率相比C0和C3均提高至少0.5个百分点。
5. 不增加推理模型参数或模块。
6. 所有正确性检查通过。

同时报告配对CI和实际损失差异。

若点估计达到门槛但CI跨越0，只报告值得进一步研究的先导信号，不宣称统计显著。

如果C3已经获得与M3相当或更好的性能，则不宣称逐层方法具有独立必要性。

## 十四、资源预算

本轮需要用户新授权。

建议：

- 一张符合实验室共享管理规定的RTX4090。
- GPU总占用上限30分钟。
- CPU累计计算上限45分钟。
- 梯度训练轮数0。
- 仅使用已有模型。

先使用少量校准样本进行统计耗时测速。

若预计完整逐层校准搜索超出预算，优先完成EXP-009及EXP-010A，并按固定候选顺序执行能完成的C3/M3配置。

不能根据已观察的验证性能选择计算资源分配。

禁止自动扩大搜索范围。

逐层BN校准额外前向较多，应完整记录计算成本。

使用可靠预算和退出状态记录机制。

## 十五、科研状态更新

新增research/ROUND-010-TASK.md。

THEORY.md追加v0.1.6及R28—R31，明确代数推导、理论假设与候选算法之间的区别。

HYPOTHESES.md：

- H-007：CURRENT_EVIDENCE_AGAINST，仅针对原微调回升预测；预算约束效应不可识别。
- H-008：INSUFFICIENT_EVIDENCE。
- H-009：UNTESTED。
- H-010：UNTESTED。

保留H-001至H-006历史科研状态。

EXP-008：COMPLETED，筛选失败。

新增EXP-009、EXP-010。

DECISION_LOG.md追加DEC-010：

暂缓额外微调，利用已有检查点开展预算匹配评价及归一化感知活动剪枝先导实验。

LITERATURE.md补充：

- EagleEye: Fast Sub-Net Evaluation for Efficient Neural Network Pruning. ECCV 2020. https://arxiv.org/abs/2007.02491
- Rethinking "Batch" in BatchNorm. Wu and Johnson, 2021. https://arxiv.org/abs/2105.07576
- Membrane Potential Batch Normalization for Spiking Neural Networks. ICCV 2023.
- Spiking Neural Networks Calibration. ICML 2021. https://proceedings.mlr.press/v139/li21d.html

说明普通BN统计量重新校准已存在相关研究，本轮尚未证明独立创新性。

## 十六、结果交付

建立独立EXP-009、EXP-010目录及唯一Run ID。

至少保存：

- TASK.md
- config.yaml
- environment.txt
- code_version.json
- commands.sh
- status.json
- candidate_history.csv
- selected_configs.json
- selection_lock.json
- bn_statistics.csv或等价可复算数据
- bn_mode_comparison.csv
- validation_per_sample.csv
- validation_batch_sops.csv
- layer_metrics.csv
- summary.csv
- metrics.json
- run.log
- REPORT.md
- limitations.md
- figures/

至少提供以下图表：

1. 不同权重状态的Accuracy–SOPs权衡。
2. 普通BN校准与逐层BN校准对比。
3. 同预算下的准确率和任务损失。
4. 每层BN统计量变化。
5. BN模式切换引起的逐层FR变化。
6. 校准计算成本对比。

未运行实验保持NA。

禁止生成虚构数据。

## 十七、GitHub版本及交付协议

从ROUND-009最终结果SHA创建独立分支。

正式运行前提交预注册任务、配置、代码和检查点来源。

记录真实执行SHA。

运行期间禁止改变执行代码HEAD。

完成后同步所有真实原始数据、报告、状态记录和必要代码。

不修改main、不force push、不上传大型模型权重和敏感信息。

最终汇报：

- GitHub实验分支。
- 执行代码SHA。
- 结果提交SHA。
- EXP-009及EXP-010执行状态。
- 20%及30%目标下各组真实结果。
- 实际预算匹配情况。
- 已有微调模型是否优于简单阈值校准。
- BN模式诊断结果。
- C3和M3的实际准确率及SOPs。
- 配对置信区间。
- 是否达到H-009及H-010预设门槛。
- 实际计算资源消耗。
- 全部异常及局限。
- 原始数据和报告路径。

本轮不追求增加算法复杂度，而是检验现有训练投入能否产生同预算收益，并验证归一化校准是否值得成为后续活动剪枝方法的组成部分。

获得用户资源授权且所有必要前置检查通过后，直接开展正式实验。
