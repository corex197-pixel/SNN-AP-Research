# Codex 科研执行任务：ROUND-021

# UniSparse脉冲暴露感知权重稀疏训练先导实验

## 1. 任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-021
- 科研决策：DEC-021，STOP + PIVOT
- 理论版本：v0.2.7
- 科研阶段：权重剪枝方向的理论诊断与真实短程训练
- 新科学问题：Q-011，实际脉冲暴露与权重稀疏惩罚分配
- 新科研假设：H-022、H-023
- 实验编号：EXP-023
- 候选方法：SEAWP，Spike-Exposure-Aware Weight Pruning
- 项目仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-020-task-protected-fanout
- 上轮执行代码SHA：806e44a7c11da60da20b6266b74a8a7521e2b490
- 上轮最终结果SHA：a4955207c719bcfdf9c7ebe62070915f9659c129
- 本轮建议分支：research/round-021-spike-exposure-pruning
- 官方UniSparse固定SHA：b5b3da4fcd2e7abcd2c20c93d41cf63ef3201daa
- 依赖实验：EXP-017、EXP-018、EXP-019、EXP-020、EXP-021、EXP-022

P0：旧方向归档、原权重剪枝理论及实际实现核验。

P0：真实脉冲暴露差异诊断。

P1：条件性执行三组各10轮真实短程训练。

P2：结果统计、资源审计及科研交付。

本轮不重新训练完整Baseline，不继续MP-EFR/TP-EFR，不修改神经元动力学。

## 2. ROUND-020正式科研结论

EXP-022为COMPLETED。

整体推进门槛失败。

真实任务梯度冲突存在：

static_conv.norm：

G0余弦=0.1930277582。

G1余弦=0.2571520410。

conv1.norm：

G0余弦=-0.1518636198。

G1余弦=-0.1554194938。

H-020更新为SUPPORTED，严格限定当前模型、数据和替代梯度协议。

梯度投影约束成立，但不保证真实硬脉冲任务损失下降。

q=0.02时：

P1相对G1实际CE增加0.0015082431。

P0相对G0实际CE减少0.0018276538。

固定验证结果：

B0：94.2444%，86.008932M SOP。

G0：94.3333%，82.163161M SOP。

G1：94.2889%，82.161247M SOP。

P0：94.1111%，82.160941M SOP。

P1：94.0667%，82.171241M SOP。

P1-G0准确率差=-0.2667个百分点。

P1-G1=-0.2222个百分点。

P1-P0=-0.0444个百分点。

三项均未达到+0.3个百分点门槛。

准确率2×2交互项为0，95%CI跨零。

H-021更新为CONTRADICTED，严格限定预注册固定模型局部干预性能预测，不推广为所有训练条件均无效。

H-017保持INSUFFICIENT_EVIDENCE，记录当前方法未通过性能筛选。

H-016、H-018、H-019保留此前限定科学状态。

正式停止MP-EFR/TP-EFR主线，不开展额外修补或正式训练。

归档全部负面结果。

## 3. 新研究方向

保留UniSparse Controlled Baseline。

核心研究问题：

UniSparse使用空间形状相关系数控制权重软阈值增长，但SNN实际单连接计算代价同时取决于输入脉冲活动。

本轮验证能否在保持原作者活动正则化、权重软阈值机制及总阈值增长系数约束的前提下，通过脉冲暴露信息重新分配层间权重稀疏化强度。

本轮只是先导研究。

不能将原作者没有直接使用实时活动代价描述为作者完全没有认识到连接与活动耦合问题。

不能将已有的权重重新生长机制描述为新创新。

## 4. 原始代码审查

读取：

vendor/UniSparse/models/submodules/layers.py

vendor/UniSparse/utils/utils.py

vendor/UniSparse/main.py

experiments/ROUND-016/

experiments/ROUND-017/

确认实际：

SparseConv2d.sparse_weight()

SparseConv2d.sparse_mask()

PseudoRelu.backward()

SparseConv2d.penalize()

SparseBN.penalize()

calc_penalty()

penalize()

原权重公式R90：

W_eff=sign(W)*relu(abs(W)-d)

R91：

M=I(abs(W)>d)

实际掩码以官方sparse_mask()为准，核查边界及精度。

R92：

d_next=d+lr*lambda_weight*K_l

官方PseudoRelu在前向使用ReLU，反向直接传递grad_output。

保留潜在权重的正常任务梯度及重新激活机制。

不引入新的STE，不改变官方神经元、任务损失、稀疏阈值参数化。

## 5. 父检查点

固定使用EXP-018的U2第100轮检查点。

预期路径：

experiments/EXP-018/runs/EXP-018-CONTINUE-001/U2/epoch-100.pth

实际路径及SHA256必须读取EXP-018的checkpoint_manifest.json确认。

不得虚构检查点哈希。

若该检查点不存在或无法可靠恢复，停止本轮训练，报告阻塞。

不得自动替换为第200轮、第300轮或best_validation检查点。

父检查点保持只读，运行前后核对SHA256。

所有训练组从同一父状态独立开始。

恢复模型、optimizer、Momentum、scheduler、AMP scaler、稀疏buffer及RNG状态。

完整Cosine周期仍为300轮。

本轮训练固定为第101—110轮。

## 6. 固定训练协议

继承EXP-018原训练配置：

CIFAR-10。

官方CifarNet。

base_channels=256。

T=4。

官方CuPy LIF。

tau=2。

v_threshold=1。

v_reset=0。

detach_reset=True。

ATan替代梯度。

batch_size=256。

workers=4。

SGD。

Momentum=0.9。

weight_decay=0。

AMP=True。

accumulation_steps=1。

seed=12450。

初始学习率0.1。

Cosine周期300轮。

lambda_weight=8.834108737251828e-08。

lambda_neuron=8.834108737251828e-08。

保留原训练数据增强、归一化、任务损失及作者AMP跳步后的手动稀疏更新顺序。

不重置scheduler。

不改变lambda_weight或lambda_neuron。

不训练超过第110轮。

训练配置以EXP-018真实解析记录核对，出现差异必须记录。

## 7. 数据协议

继续使用：

experiments/EXP-004/splits/indices.json

SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

训练集45000张。

验证集4500张。

旧500张诊断池不使用。

禁止读取官方test_batch。

训练顺序及增强随机控制继承EXP-018。

每轮175个Batch。

三组对应Epoch的采样顺序及尾批丢弃索引必须一致。

暴露诊断复用EXP-020 D_grad的512张训练来源样本。

另用EXP-020 D_cal的512张训练来源样本做暴露估计稳定性检查。

实际索引直接读取历史data_split.json。

不重新搜索样本。

确定性验证预处理。

两组样本均已参加父模型训练，因此不能称为独立未见样本。

## 8. EXP-023A：真实脉冲暴露测量

使用第100轮固定父模型。

在eval、no_grad、FP32模式下使用真实消费者输入。

测量实际接受权重惩罚且位于SOP监控范围的全部SparseConv2d。

包括：

conv1.conv。

conv2。

conv3.conv。

conv4.conv。

conv5。

fc1。

fc2。

实际模型名称和监控范围以源码核查为准。

static_conv若原K_l=0，则保持未惩罚。

输入：

x.shape=[T,B,C,H,W]

M.shape=[C_out,C_in,k_h,k_w]

对于卷积核位置(a,b)，按实际stride、padding、dilation映射：

i_h=u*stride_h-padding_h+a*dilation_h

i_w=v*stride_w-padding_w+b*dilation_w

越界输入为0。

R93：

e[c,a,b]=sum_{n,t,u,v} x[t,n,c,i_h,i_w]/(B*T)

其中B为整个实际诊断样本数量，不是固定单Batch大小。

计算时按所有样本累计原始计数后统一归一化。

e表示平均每样本每时间步的有效输入脉冲暴露次数。

R94：

C_l=T*sum_{j,c,a,b} M[j,c,a,b]*e[c,a,b]

R95：

固定输入活动时，删除一个有效权重的直接局部SOP节省=T*e[c,a,b]。

上述公式要求groups=1及实际输入为二值脉冲。

如存在其他输入语义，报告不适用范围，不强行套用。

可使用卷积展开、按位置切片或等价的高效统计方式。

不得为了诊断构造导致显存溢出的巨大完整unfold张量。

## 9. 暴露比例R96

对相同形状的全1输入，使用完全相同的卷积空间映射计算e_dense。

R96：

p_l=sum(e_l)/sum(e_dense_l)

要求：

0<=p_l<=1

分母必须为正且有限。

如果分母为0，报告无效层及原因。

本轮p_l为所有潜在输入连接位置的平均暴露比例，不以当前掩码筛选高活动连接。

记录：

- 实际输入shape。
- 卷积参数。
- 全局FR。
- p_l。
- e的均值、标准差及分位数。
- 原K_l。
- 原权重阈值。
- 有效连接密度。
- 实际层SOP。
- R94重构SOP。
- 与官方Monitor的误差。

池化、flatten后的实际消费者输入直接测量，不采用源神经元活动的近似替代。

## 10. 正确性检查

至少完成：

1. CPU人工整数卷积暴露与显式枚举一致。
2. padding、stride及dilation处理正确。
3. 全1输入p_l=1。
4. 全0输入p_l=0。
5. 固定掩码下R94与实际连接枚举一致。
6. 单连接删除成本差与R95一致。
7. 非完整尾批的样本加权正确。
8. 真实网络R94与原SOPMonitor一致。
9. 父模型状态未被诊断修改。
10. 捕获开启和关闭不改变模型输出。

真实网络建议使用归一化误差不超过1e-5的SOP重建检查。

不得通过扩大容差隐藏公式错误。

## 11. H-022及进入训练门槛

H-022：

相同原K_l的层之间存在具有实际幅度的真实脉冲暴露差异。

预先指定主要层集合：

conv1.conv、conv2、conv3.conv。

这三层历史原K_l均为1024。

使用D_grad测得p_l。

定义：

A_p=(max(p_l)-min(p_l))/mean(p_l)

若mean(p_l)=0，标记无效，不除零。

探索性进入门槛：

A_p>=0.20。

且D_cal上A_p同样>=0.20。

两组数据中最大和最小暴露层的身份必须一致。

此门槛只是判断是否存在值得继续研究的实际层间异质性，不是理论必然阈值。

若门槛未通过：

停止EXP-023后续训练。

报告H-022当前主要条件下未获得支持。

不更换主要层，不改变阈值，不寻找有利样本。

若通过：

按已授权协议进入三组各10轮短程训练。

## 12. 新惩罚公式R97—R100

令L为所有同时满足：

- 原K_l>0。
- 输入为已验证二值活动。
- 暴露统计正确。
- 原权重惩罚适用。

的SparseConv2d集合。

本轮若预期主要受惩罚层不满足条件，应先报告，不能未经说明随意删减研究层。

R97：

p_bar_K=sum_l(K_l*p_l)/sum_l(K_l)

R98：

K_hat_l=K_l*(1-alpha+alpha*p_l/p_bar_K)

固定alpha=0.25。

若所有p_l=0导致p_bar_K=0，回退K_hat=K并记录方法退化；不进行重复无效训练。

R99：

d_next_l=d_l+lr*lambda_weight*K_hat_l

R100：

sum_l(K_hat_l)=sum_l(K_l)

对于alpha=0，应严格恢复原方法。

对于所有p_l相等，应恢复原方法。

K_hat必须有限且非负。

使用float64计算归一化系数，再按照实际buffer精度应用。

记录实际float32累计误差。

不得使用额外epsilon改变公式含义。

本轮不使用动态EMA。

使用父模型D_grad暴露比例计算固定K_hat，训练10轮期间不更新该系数。

D_cal仅用于诊断稳定性，不用于选择p_l、alpha或系数。

## 13. 实验分组

W0：

原始K_l。

W1：

按照R97—R100计算真实暴露感知K_hat_l。

W2：

将D_grad暴露比例p_l按照固定层顺序随机置换，再执行完全相同的R97—R100。

固定CPU随机种子20261128。

置换层集合、排列顺序及实际permutation必须保存。

不得重新选择置换。

注意：

置换后的p_l多重集合保持不变，但由于K_l不同，归一化后K_hat多重集合不一定保持不变。

W2是暴露信息对应关系对照，不是严格相同惩罚系数分布对照。

应明确报告该局限。

三组保持sum(K_hat_l)=sum(K_l)。

未参与修改的层保留原K_l。

如果固定置换与原暴露分配完全相同或实质退化，报告W2无法形成有效对照，不自行更换随机种子。

## 14. 实现范围及重要约束

不修改vendor/UniSparse官方代码。

优先在独立ROUND-021训练入口实现权重惩罚系数适配。

必须保留作者calc_penalty计算活动惩罚的原始逻辑。

特别注意：

原calc_penalty同时计算权重形状惩罚和BN活动惩罚。

本轮只允许在应用权重阈值更新之前修改对应SparseConv2d的权重penalty系数。

禁止改变：

SparseBN.penalty_gamma。

SparseBN.penalty_beta。

SparseBN手动更新方式。

神经元活动梯度。

原任务反向传播。

卷积权重STE。

学习率及优化器。

建议流程：

执行原任务前向、反向和优化器更新。

执行原calc_penalty。

核对原始K_l。

仅对W1/W2指定权重层应用固定K_hat_l。

执行原penalize。

执行原clear_penalty。

实际更新顺序与EXP-018一致。

若原K_l在本轮训练条件下并非常数，暂停当前实现，报告理论适用条件问题。

## 15. 训练前置检查

必须验证：

- 父检查点哈希。
- 环境、数据及代码版本。
- 模型、Momentum、scheduler和scaler恢复。
- 下一轮实际学习率。
- 三组初始参数完全一致。
- 三组固定lambda一致。
- W0更新行为与原EXP-018一致。
- 原K_l与历史记录一致。
- R100成立。
- W1/W2确实产生不同的实际阈值增量。
- 三组同一临时Batch的任务前向在稀疏更新前一致。
- 同一临时Batch的BN活动惩罚计算不被新增系数直接修改。
- 原始潜在权重仍能获得任务梯度。
- AMP及非有限状态处理符合原协议。

注意：

后续模型状态不同会自然导致BN活动惩罚梯度不同。只要求活动惩罚算法不变，不要求整个训练期间活动梯度数值相同。

临时测试模型不得用于正式训练。

正式训练从原父检查点独立恢复。

## 16. 正式短程训练

仅在H-022进入门槛、正确性检查和用户授权全部满足后执行。

W0、W1、W2各训练10个Epoch。

固定第101—110轮。

相同训练预算。

固定第110轮作为主要评价点。

第105轮作为预先规定的中间观察点。

不得按验证准确率选择更有利Epoch作为主要结果。

禁止额外训练、微调或调整alpha。

每轮保存可恢复最新检查点。

保存第105、110轮固定检查点。

若训练异常中断，保留所有完整Epoch。

不得把未完成训练组与完整10轮组进行主要比较。

## 17. 训练记录

每轮保存：

- 训练损失和准确率。
- 验证损失和准确率。
- 实际学习率。
- SOP_author。
- SOP_sample_weighted。
- 全局及逐层FR。
- 全部稀疏层权重稀疏率。
- 受惩罚层权重稀疏率。
- 非零有效权重总量。
- 各层有效连接密度。
- 原K_l及实际K_hat_l。
- 各层阈值。
- 原始权重绝对值分位数。
- 成功优化器更新数。
- AMP跳步数。
- 实际手动稀疏更新次数。
- GPU/CPU资源。
- 样本顺序及丢弃样本哈希。

在第100、105、110轮复测实际p_l。

训练中复测的暴露值只用于描述性诊断，不用于改变K_hat。

记录训练过程中暴露变化与固定分配系数之间可能出现的失配。

## 18. 权重拓扑变化诊断

在每个固定检查点保存或流式统计有效掩码。

相对于父模型掩码M100，计算：

pruned_count=sum(M100=1 and M_current=0)

regrown_count=sum(M100=0 and M_current=1)

分别按层报告。

最终连接数应满足：

N_current=N_parent-pruned_count+regrown_count

该统计用于描述原UniSparse已有的潜在权重再激活机制及候选方法下的拓扑变化。

不得将原始再生能力宣称为新创新。

掩码统计不得改变实际训练状态。

## 19. 主要评价指标及推进门槛

固定第110轮。

W0为主要Baseline。

W2为信息对应关系对照。

H-023为探索性性能假设。

W1必须满足以下任一预先规定的推进路径，且对W0和W2均满足同一路径。

路径A，计算效率：

- SOP相对对应对照下降至少5%。
- 准确率相对对应对照下降不超过0.3个百分点。

路径B，任务性能：

- SOP不超过对应对照的101%。
- 准确率相对对应对照提高至少0.3个百分点。

此外：

- 全部科学和工程检查有效。
- 训练轮数一致。
- 关键受惩罚层有效连接密度均至少1%。
- 参数、损失及手动更新保持有限。
- 无数据泄漏或未授权配置改变。

报告实际权重稀疏率变化及非零连接数量。

不得将SOP下降自动等同于参数压缩改善。

不得把不同SOP条件下的精度差描述为严格同预算比较。

若两条路径均失败，不追加训练、不调整alpha、不改变分组，返回理论研究员审查。

这些是探索性推进门槛，不是统计显著性判据。

## 20. 配对统计

使用4500张固定验证样本。

按样本ID和标签配对。

主要比较：

W1-W0。

W1-W2。

固定2000次配对Bootstrap。

seed=20261128。

numpy.default_rng PCG64。

报告准确率百分点差、CE差及95%百分位区间。

如果能够正确采集逐样本SOP，则报告成本差及配对区间；否则标记NA。

保存各组真实逐样本预测。

当前仅一个父训练种子，不能宣称跨训练种子稳定性。

验证集已多次用于开发，不属于独立确认性测试。

## 21. 资源预算

需要用户新授权。

建议：

一张符合实验室共享管理规定的RTX4090。

GPU总占用上限45分钟。

CPU累计计算上限90分钟。

正式新增训练最多30个Epoch。

优先选择实际可用显存不少于18 GiB的GPU。

所有前置检查、失败重试、暴露诊断、训练及最终评价计入预算。

禁止使用第二张GPU。

不得干扰其他用户的进程。

不得因进程重启而重置累计预算。

使用已验收的持久化训练、预算监控和原子检查点机制。

## 22. 停止条件

出现以下情况停止受影响任务：

- H-022进入门槛失败。
- 父检查点缺失或哈希不一致。
- 原始代码版本不一致。
- 暴露计数与Monitor不匹配。
- 权重惩罚之外的科学算法被意外改变。
- R100或实际阈值更新不正确。
- 三组采样及训练配置不一致。
- 非有限参数、损失或更新。
- OOM。
- 关键层连接耗尽。
- GPU/CPU预算超限。

正常AMP跳步按原协议记录。

不擅自调整Batch Size、lambda、学习率、alpha或训练目标。

## 23. 文献与创新性记录

更新LITERATURE.md。

重点保留：

UniSparse：

https://proceedings.mlr.press/v306/shi26a.html

Spiking Activity Based Pruning：

https://proceedings.mlr.press/v235/li24bz.html

Dynamic Joint Pruning Framework：

https://ieeexplore.ieee.org/document/11132570/

Unsupervised Adaptive Weight Pruning：

https://doi.org/10.3389/fnins.2020.598876

当前候选与活动感知阈值调整、动态联合剪枝存在潜在重复风险。

不得宣称SEAWP为学术首创。

R93—R95属于固定活动下的卷积SOP计数。

R98为待验证的层级分配设计。

若先导通过，再由理论研究员开展更严格的公式级查重。

## 24. 科研状态更新

新增：

research/ROUND-021-TASK.md

更新：

THEORY.md，追加v0.2.7及R90—R100。

HYPOTHESES.md，正式更新H-020/H-021，新增H-022/H-023。

RESEARCH_STATE.md。

EXPERIMENT_PLAN.md。

EXPERIMENT_REGISTRY.csv。

DECISION_LOG.md，追加DEC-021。

LITERATURE.md。

保留EXP-020、021、022为COMPLETED及原失败门槛。

MP-EFR/TP-EFR标记STOPPED或AGENTS.md适用状态。

UniSparse Controlled Baseline保留。

不删除或改写任何历史负面证据。

## 25. 实验交付

建立独立：

experiments/EXP-023/

experiments/ROUND-021/

至少保存：

TASK.md

config.json

environment.txt

code_version.json

commands.sh

status.json

parent_checkpoint_audit.json

baseline_formula_audit.md

exposure_formula_tests.json

exposure_statistics.csv

exposure_stability.csv

sop_reconstruction_audit.json

weight_penalty_coefficients.csv

permutation.json

selection_lock.json

preflight_tests.json

hypothesis_gate_results.json

epoch_metrics.csv

layer_sparsity.csv

threshold_trajectory.csv

topology_transition.csv

training_sample_audit.csv

validation_per_sample.csv

validation_summary.csv

paired_bootstrap.json

checkpoint_manifest.json

resource_usage.csv

REPORT.md

limitations.md

run.log

figures/

未运行部分标记NA并说明原因。

不虚构训练结果。

保持唯一Run ID，保留失败记录。

图表至少包括：

逐层实际脉冲暴露比例。

原始K_l与新K_hat_l对比。

训练准确率及损失曲线。

SOP随训练轮次变化。

逐层权重稀疏率变化。

连接剪除与重新激活统计。

固定第110轮精度、SOP及权重稀疏率比较。

## 26. GitHub交付及最终反馈

从上轮最终提交创建独立分支。

正式GPU任务开始前提交任务、配置、门槛及实现。

进入训练前提交实际暴露统计、系数、置换和selection_lock。

实验结束后提交全部真实结果和状态文件。

不修改main，不force push，不覆盖历史实验。

大型权重保留服务器，记录真实路径及SHA256。

最终返回：

实验分支链接。

实际执行代码SHA。

最终结果提交SHA。

EXP-023执行状态。

父检查点路径及SHA256。

H-022实际诊断结果。

各层p_l及K_hat_l。

所有正确性检查结果。

三组实际完成训练轮数。

固定第110轮Accuracy、CE、FR、SOP、权重稀疏率。

逐层连接密度、剪除及再激活统计。

配对比较及置信区间。

推进门槛结果。

真实资源消耗。

全部异常、限制及未完成项目。

实验报告和原始数据路径。

本轮旨在验证新的权重稀疏训练机制，不再修补已经停止的MP-EFR/TP-EFR方法。

获得用户授权后执行。