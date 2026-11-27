# Codex 科研执行任务：ROUND-013

# AT-LIF 结构化通道剪枝及任务重要性先导验证

## 1. 任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-013
- 科研阶段：新方向低成本缺陷及可行性验证
- 科研决策：DEC-013，STOP + PIVOT
- 理论版本：v0.1.9
- 科学问题：Q-009
- 科研假设：H-014、H-015
- 实验编号：EXP-014
- GitHub仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-012-effective-threshold-gradient
- 本轮建议分支：research/round-013-structured-channel-pruning
- EXP-012执行代码SHA：7ca11f85d51c67b874158906819611bd88f0694e
- EXP-013执行代码SHA：071b5fb6f1dc8b9ff172d45fe13434ab7597665b
- 上轮最终结果SHA：604060bc17aada022579dba15dac19c4e9845b12
- 依赖实验：EXP-004、EXP-007、EXP-012、EXP-013
- 优先级：P0结构依赖及删除等价性检查；P1通道重要性测量；P2真实结构化剪枝评价

首先读取AGENTS.md、BASELINE.md、THEORY.md、HYPOTHESES.md、RESEARCH_STATE.md、EXPERIMENT_PLAN.md及ROUND-012实验报告。

从上轮结果SHA建立独立实验分支。

## 2. ROUND-012正式科研判定

EXP-012及EXP-013均为COMPLETED。

H-012：CURRENT_EVIDENCE_SUPPORTS，限定当前模型、固定检查点、数据及SGD协议。

EXP-012：

- D0：16次成功更新，0/19层发生非零优化器阈值更新。
- G0-S、G1-S、M4-S：各16次成功更新，19/19层发生非零优化器阈值更新。
- 独立审计1216条层级更新，最大误差0 ULP。
- H-012工程干预门槛通过。

EXP-013：

20%主要预算：

- C0：Accuracy=0.908，SOPs=43652928.192。
- G0-S：Accuracy=0.9042222222，SOPs=43555934.7556。
- G1-S：Accuracy=0.9015555556，SOPs=43607919.7369。
- M4-S：Accuracy=0.9024444444，SOPs=43417726.7627。

M4-S相对C0准确率差为-0.5556个百分点。

M4-S相对G0-S准确率差为-0.1778个百分点。

三组均完成5轮训练，最终模型两两不同。

30%辅助预算：

- M4-S相对C0准确率提高0.8889个百分点。
- M4-S相对G0-S仅提高0.0667个百分点。
- 不能认定ARTG具有独有优势。

H-013：INSUFFICIENT_EVIDENCE，原主要性能筛选失败。

保留所有历史结果，包括30%预算探索性正向信号。

正式停止推进当前ARTG方案。

梯度尺度平衡保留为备选研究问题，不继续开展梯度倍率搜索或完整200轮训练。

## 3. 新研究目标

已有AT-LIF主要压缩脉冲活动，模型参数量始终为1084605。

本轮转向真实结构化通道剪枝。

需要回答：

1. 已训练的AT-LIF是否仍存在可低损失删除的内部结构冗余？
2. 低发放率是否能够可靠描述通道的重要性？
3. 直接测量任务损失变化，能否获得优于简单通道指标的选择结果？

本轮仅属于低成本可行性及机制探索。

不预先宣称通道剪枝、直接损失敏感性或活动与结构联合压缩具有原创性。

## 4. 固定模型

使用EXP-004原始第194轮最佳检查点：

experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth

SHA256：

c068f54379cb5d29bcfa5fe2d9de676cd3cc153433ada1ed5aa1c66355d0b32a

使用原始阈值，不乘1.25。

不使用任何ROUND-009、ROUND-011或ROUND-012微调检查点。

固定：

- CIFAR-10。
- ResNet-20。
- 原AT-LIF。
- T=8。
- tau=0.9。
- FP32。
- model.eval()。
- torch.no_grad()。
- Batch Size=128。
- num_workers=4。
- 原确定性评估预处理。
- 禁止梯度训练。
- 禁止手动阈值更新。
- 禁止BN统计量重新校准。

原检查点只读，不得覆盖。

## 5. 数据协议

原固定数据划分：

experiments/EXP-004/splits/indices.json

SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

使用现有500张训练来源校准样本：

experiments/EXP-007/calibration_indices.json

SHA256：

b849b74ef19cb4a09df756c54871122f3b02aa047c704d0436feab15a38bde52

使用独立NumPy RNG，seed=2031，将原500张校准样本按类别分层固定划分：

- A：250张，每类25张，用于计算剪枝重要性及确定最终剪枝通道。
- B：250张，每类25张，仅用于评价重要性排序的样本稳定性。

保存A、B的官方训练集原始索引、随机生成算法及SHA256。

A、B不得重叠。

B不得用于调整已由A确定的剪枝通道。

最终性能使用原4500张验证集。

必须在任何验证集评价之前锁定所有剪枝组的结构及通道索引。

原500张科研诊断池不使用。

官方10000张测试集继续隔离。

说明：A、B虽然样本互斥，但都来自曾参与原模型训练及之前校准的样本，因此只能用于内部稳定性分析，不能当作全新独立泛化数据。

## 6. 结构化剪枝对象

实际源代码：

Activity-Pruning-SNN-main/models/resnet_cifar.py

预先指定三个目标模块：

- conv2_x.1.residual_function.2
- conv3_x.1.residual_function.2
- conv4_x.1.residual_function.2

预期对应内部通道数：

32、64、128。

首先核查真实网络中上述模块的名称、类型、输入输出形状及消费者关系。

每个目标模块预期处于以下路径：

Conv1 -> BN1 -> ATLIF -> Conv2 -> BN2

仅删除该内部ATLIF对应的通道。

保持残差块外部输入输出尺寸不变。

不修改Shortcut。

不删除残差相加之后的ATLIF通道。

必须确认目标ATLIF输出只被预期的第二个卷积消费。

如果实际计算图存在其他消费者，暂停对应结构删除并报告，不擅自扩大剪枝依赖范围。

## 7. 真实结构删除公式

设保留通道索引为K。

第一层卷积：

W1_new = W1[K,:,:,:]

BN1：

- weight_new = weight[K]
- bias_new = bias[K]
- running_mean_new = running_mean[K]
- running_var_new = running_var[K]

保持原BN eps、momentum、eval状态及必要计数buffer。

第二层卷积：

W2_new = W2[:,K,:,:]

保持输出通道数不变。

ATLIF共享标量阈值、T、tau及原神经元机制不变。

使用与实际原卷积及BN等价的SpikingJelly多时间步模块。

保持kernel_size、stride、padding、dilation、groups、dtype和device。

本轮目标模块预期为groups=1且无卷积偏置；若实际配置不符，先报告并核查公式适用性。

R47：

delta_parameters_per_channel = k²*(C_in+C_out)+2

理论预计：

- 删除8个32通道阶段内部通道。
- 删除16个64通道阶段内部通道。
- 删除32个128通道阶段内部通道。
- 总计56个内部通道。
- 参数减少96880。

这些数值必须通过实际模型参数计数独立验证。

不得仅将权重置零后报告参数量下降。

## 8. 通道重要性定义

### R48：发放率重要性

对于实际ATLIF输出o，且theta严格为正：

s = (o > 0)

逻辑维度[T,B,C,H,W]。

对每个通道：

I_FR[c] = sum_{t,b,h,w}(s[t,b,c,h,w]) / 实际状态元素总数

跨Batch按实际元素数量累计。

不得简单平均不同大小Batch的发放率。

### R49：滤波器L1重要性

I_L1[c] = sum(abs(W1[c,:,:,:]))

使用第一层卷积实际权重。

仅在同一个目标模块内部进行通道排序，因此本轮不额外引入跨层归一化。

### R50：实际删除损失

对于每个目标通道c：

在ATLIF输出之后、Conv2输入之前，将该通道的完整[T,B,H,W]输出设为0。

保持其他全部通道及参数不变。

定义：

D_c = mean_b(CE(masked_logits[b],label[b]) - CE(reference_logits[b],label[b]))

任务logits使用原模型：

logits = mean_t(model_output)

必须在完整网络真实前向中获得损失。

不使用替代梯度或局部输出距离作为D_c。

所有单通道候选从同一个未修改的原始模型开始。

删除候选之间不得累积掩码。

允许D_c为负，不进行截断。

剪枝时优先删除D_c最小的通道。

D_c为直接干预参考指标，不宣称为新发明的剪枝准则。

## 9. 校准及排序稳定性实验

在A集上计算全部224个目标通道的：

- I_FR。
- I_L1。
- D_c。

在B集上独立测量相同指标。

不得根据B集结果重新选择通道。

计算每个目标模块：

- Spearman(I_FR_A,D_B)。
- Spearman(I_L1,D_B)。
- Spearman(D_A,D_B)。
- A集最低25%发放率通道的D_B分布。
- A集最低25%实际损失通道的D_B分布。
- A、B直接损失排序的最低25%通道集合重合率。

Spearman使用平均秩处理并列值。

常数指标导致相关系数未定义时保存NA及原因，不填0。

保存全部通道候选，包括负损失增量、静默通道和异常通道。

这些是探索性机制统计，不以事后选择的相关系数阈值作为预注册科学证明。

## 10. 实验组

B0：

原始AT-LIF，无结构删除。

P-Random：

固定独立RNG，seed=2032，在每个目标模块均匀无放回随机选择25%通道删除。

P-L1：

删除每个目标模块I_L1最小的25%通道。

P-FR：

删除每个目标模块I_FR最小的25%通道。

P-Loss：

删除每个目标模块D_c最小的25%通道。

所有排序仅使用A集及原模型权重。

数值并列时按原始通道索引升序选择。

三个目标模块的删除数量固定为8、16、32。

所有剪枝组必须独立从原模型构建。

不得从某个已剪枝组继续构造其他组。

不进行迭代重新排序，不添加新候选方法。

## 11. 结构删除等价性检查

正式评价之前必须完成。

### 11.1 人工结构测试

构造小型Conv-BN-ATLIF-Conv模块。

比较：

A：保持原结构，在ATLIF输出后使用通道掩码。

B：实际删除对应卷积和BN通道。

在eval模式下检查：

- 保留通道输出。
- 最终卷积输出。
- AT-LIF阈值。
- 参数计数。
- 输出形状。
- 残差相加兼容性。

### 11.2 真实网络测试

从A集预先固定前32张图片。

比较各最终剪枝配置的：

- 原模型加对应多通道掩码。
- 真实紧凑网络。

比较完整时间平均logits及预测类别。

float32归一化logits误差预设容差1e-5：

max(abs(z_compact-z_masked)/(1+abs(z_masked)))

同时保存最大绝对误差、预测分歧数、发生脉冲差异的模块及原因。

硬阈值附近的微小卷积舍入误差可能改变发放轨迹。不能仅凭理论等价性就假定实际浮点结果必然逐位一致。

若存在超过容差的差异，暂停相关正式评价，查明结构依赖错误或数值边界问题。

不得事后放宽容差以掩盖实现错误。

### 11.3 状态与资源检查

- 原检查点哈希不变。
- 保留通道权重和BN状态正确。
- 未删除参数无未经指定的修改。
- ATLIF状态正确清理。
- 实际紧凑模型张量尺寸缩小。
- 全部剪枝组参数量相同。
- 无NaN或Inf。
- 无未经批准的训练。

## 12. SOPs及参数量评价

使用原作者SOPMonitor的相同统计口径：

Activity-Pruning-SNN-main/utils/utils.py

对真正缩小后的模型重新注册监控器。

根据实际卷积权重形状统计理论活动运算量。

注意：

仅使用激活掩码的原尺寸网络，第一层卷积仍计算全部输出通道，因此其SOPs不能替代真实紧凑结构的SOPs。

分别记录：

- Masked模型的任务性能，用于等价性检查。
- Compact模型的任务性能。
- Compact模型实际参数量。
- Compact模型实际SOPs。
- 原始卷积张量尺寸及剪枝后尺寸。
- 全网络FR。
- 各目标模块局部FR。
- 相对原模型的参数减少比例。
- 相对原模型的SOPs减少比例。

不将删除通道比例25%描述为整个网络参数减少25%。

不将理论SOPs等同于硬件能耗或实际推理加速。

本轮不要求硬件能耗测量。

## 13. 正式评价及科研假设

在所有剪枝配置锁定后，使用原4500张验证集。

评价B0、P-Random、P-L1、P-FR、P-Loss全部结果。

H-014：

当前AT-LIF仍存在可低损失删除的结构冗余。

先导实用性支持条件：

至少一种真实结构化剪枝方法：

1. 达到预设的实际结构删除及参数缩减。
2. 相对B0准确率下降不超过1个百分点。
3. 实际SOPs低于B0。
4. 所有结构正确性检查通过。

H-015：

真实任务损失感知的通道选择优于简单活动及权重指标。

先导支持条件：

P-Loss相对P-FR和P-L1均提高至少0.5个百分点的准确率。

各组实际删除参数量必须相同。

同时要求P-Loss的相对B0 SOPs节省比例不比P-FR或P-L1低超过2个百分点。

若实际SOPs差异超过该范围，只报告真实精度—参数—SOPs权衡，不直接宣称同预算优势。

采用2000次样本级配对Bootstrap，seed=2027。

报告：

- 各剪枝组相对B0的准确率及CE差异。
- P-Loss相对P-FR/P-L1/P-Random的配对差异及95% CI。

点估计满足门槛但CI跨0，只报告先导信号。

注意存在多组比较，未经校正的样本级CI不能作为最终确认性证据。

若P-Loss没有优势，不自动添加微调、降低剪枝率或修改排序指标。

## 14. 新Baseline可复现性审查

在不运行额外训练的前提下，检查：

1. Towards Energy Efficient Spiking Neural Networks: An Unstructured Pruning Framework，ICLR 2024。

https://github.com/xyshi2000/Unstructured-Pruning

2. Towards Efficient Deep Spiking Neural Networks Construction with Spiking Activity based Pruning，ICML 2024。

https://proceedings.mlr.press/v235/li24bz.html

3. UniSparse: Combining Weight Pruning and Spike Sparsification in Spiking Neural Networks，ICML 2026。

https://github.com/xyshi2000/UniSparse

记录：

- 公开代码可获得性。
- 代码许可证。
- 支持的模型及数据集。
- 是否有来源可验证的训练权重。
- 依赖版本与当前环境兼容性。
- 原论文的训练预算。
- 是否支持单GPU实验。
- 实际剪枝粒度和性能评价指标。

仅作静态审查，不自动安装新环境、不下载大型数据集或启动其他Baseline训练。

在research/BASELINE_CANDIDATES.md中保存真实核查结果。

无法确认的信息写明UNKNOWN，不编造预训练权重链接。

## 15. 资源及终止条件

本轮需要用户新授权。

建议资源：

- 一张符合实验室共享管理规定的RTX4090。
- GPU总占用上限15分钟。
- CPU累计计算上限30分钟。
- 新增梯度训练0 Epoch。

优先进行结构正确性测试和少量候选计时。

按真实测速评估224个通道在两个250样本子集上的评价成本。

若预算不足，严格记录未完成状态，不根据表现挑选更容易得到正面结果的通道。

若A集全部重要性已完成而B集未完成，允许按已锁定的A集选择规则进行预算允许范围内的结构评价，但整体标记PARTIAL，明确排序稳定性诊断缺失。

发生以下情况停止相关实验：

- 检查点或数据哈希错误。
- 结构依赖关系不明。
- 掩码与真实删除等价性检查失败。
- 非有限输出。
- GPU资源不足或预算超限。
- 原模型被意外修改。

禁止额外训练、BN重新校准、梯度优化、阈值搜索、超参数搜索或使用官方测试集。

## 16. 科研状态更新

新增research/ROUND-013-TASK.md。

THEORY.md追加v0.1.9及R47—R50。

HYPOTHESES.md：

- 保留H-001至H-011。
- H-012：CURRENT_EVIDENCE_SUPPORTS，限定已验证范围。
- H-013：INSUFFICIENT_EVIDENCE，主要筛选失败。
- H-014、H-015：UNTESTED。

如果AGENTS.md的机器可读科学状态枚举使用SUPPORTED而不是CURRENT_EVIDENCE_SUPPORTS，保持明确映射，并记录适用范围，不静默修改历史结论。

EXP-012及EXP-013保持COMPLETED。

新增EXP-014。

DECISION_LOG.md追加DEC-013：

停止当前ARTG路线，保留共享阈值梯度尺度发现，转向内部通道结构化剪枝。

LITERATURE.md记录SCA、UniSparse及结构化敏感性剪枝已有研究。

不宣称本轮直接删除损失准则具有原创性。

## 17. 结果交付

在experiments/EXP-014下建立独立实验目录及唯一Run ID。

至少保存：

- TASK.md
- config.json或config.yaml
- environment.txt
- code_version.json
- commands.sh
- status.json
- calibration_split.json
- channel_dependency_map.json
- structural_equivalence_tests.json
- channel_importance_A.csv
- channel_importance_B.csv
- ranking_statistics.csv
- selected_channels.json
- selection_lock.json
- parameter_count_audit.csv
- validation_per_sample.csv
- validation_batch_sops.csv
- summary.csv
- metrics.json
- run.log
- REPORT.md
- limitations.md
- figures/

图表至少包括：

1. 各层FR与实际删除损失散点图。
2. 各层L1与实际删除损失散点图。
3. A、B实际损失排序稳定性。
4. 各方法实际参数量、准确率及SOPs。
5. 各方法删除通道索引。
6. 结构删除前后实际卷积张量尺寸。

未完成项目标记NA及真实原因。

大型模型权重留在服务器，记录路径、大小、SHA256及结构配置。

禁止上传大型数据集或敏感信息。

## 18. GitHub交付

正式执行前提交预注册任务、固定数据索引及代码。

实验完成后提交真实原始数据、报告、异常记录及科研状态。

记录实际执行代码SHA及结果提交SHA。

不修改main，不force push，不覆盖原Baseline或历史实验。

最终汇报：

- GitHub实验分支。
- 执行代码SHA。
- 最终结果提交SHA。
- EXP-014执行状态。
- 三个目标模块的实际依赖关系。
- 结构等价性检查结果。
- 每组实际删除参数量。
- 全部通道重要性统计。
- B0、P-Random、P-L1、P-FR、P-Loss的真实性能。
- H-014和H-015是否达到预设先导门槛。
- 备选Baseline可复现性审查结果。
- 实际资源消耗。
- 全部异常和局限。
- 报告及原始数据路径。

获得用户重大方向调整批准及资源授权后，执行本轮实验。

本轮目标不是强行获得一个新的创新方法，而是用尽可能少的计算资源判断SNN结构化剪枝是否更适合作为本项目下一阶段的论文主线。