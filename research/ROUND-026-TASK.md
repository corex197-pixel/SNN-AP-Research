# Codex 科研执行任务：ROUND-026

# UniSparse高比例结构化剪枝后的标准任务微调与性能恢复诊断

## 1. 任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-026
- 实验编号：EXP-028
- 科研决策：DEC-026，STOP + IMPLEMENT STRONG BASELINE
- 理论版本：v0.3.2
- 科学问题：Q-015，高比例结构化剪枝后的任务性能恢复
- 科研假设：H-028、H-029
- 项目仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-025-admission-replication
- 本轮建议分支：research/round-026-compact-finetuning
- 上轮实际执行代码SHA：303995cf1b181687f3f1c63084aa89c0a1b3c31a
- 上轮远程准入锁SHA：e620bb41bcf267e1047419cae6bc63cfab5e39c1
- 上轮最终结果SHA：37efb52b4bad7001fddd9658d63ecc5dede2f6e3
- 官方UniSparse固定SHA：b5b3da4fcd2e7abcd2c20c93d41cf63ef3201daa
- 依赖实验：EXP-018、EXP-024、EXP-026、EXP-027

P0：正式归档EXP-027及原重构方向科学判定。

P0：完成紧凑模型真实前向、反向、参数保护、状态管理及训练准入检查。

P1：执行两个固定微调组，各5轮。

P2：进行固定终点评价、配对统计、计算成本审计及科研交付。

本轮建立传统剪枝后微调强对照，不实现新的论文创新算法。

## 2. ROUND-025正式科研判定

EXP-027保持COMPLETED和VALID。

EXP-026永久保持INVALID。

H-026更新为CURRENT_EVIDENCE_AGAINST，限定原50%主要条件及至少1个百分点损失预测。

H-027整体性能假设更新为CONTRADICTED，限定固定支持集ridge重构方案。

单独记录H-027的D_cal logits重构误差下降子命题得到支持。

75%条件属于原辅助探索，不替换H-026主要条件。

固定支持集最小二乘读出重构方案标记STOPPED。

不得继续搜索lambda、重新拟合R3或实现中心化重构变体。

保留全部历史正负面科研证据。

## 3. 已验证实验依据

EXP-027严格FP32结果：

B0：

- Accuracy=94.222222%。
- CE=0.1727328626844618。
- FR=0.04957855784345042。
- SOP/sample=86009551.08977778。
- 参数量36720484。

50-R1：

- Accuracy=93.933333%。
- CE=0.1793660873969396。
- SOP/sample=78340358.21466666。
- 参数量19839844。

75-R1：

- Accuracy=92.511111%。
- CE=0.27720849974950157。
- FR=0.04940481636399951。
- SOP/sample=74488295.33955556。
- 参数量11399524。

75-R3：

- Accuracy=90.733333%。
- CE=0.32743940374586317。
- SOP/sample=74488295.33955556。

75-R1相对B0：

Accuracy差=-1.711111个百分点。

95%配对CI=[-2.311111,-1.088889]。

75-R3相对75-R1：

Accuracy差=-1.777778个百分点。

95%配对CI=[-2.533333,-1.022222]。

R121实际诊断：

D_cal原75-R1：

MSE_raw=20.662515923714174。

MSE_centered=20.62011759996856。

MSE_shift=0.042398323745617236。

公共偏移占比较小，不支持将其解释为当前重构失败的主要原因。

EXP-027与EXP-026为同一模型重新评价，不属于独立训练种子重复实验。

## 4. 新理论记录

追加v0.3.2。

R123：

教师预测类别c具有唯一最大logit。

gamma=z[c]-max_{j!=c}(z[j])。

剪枝扰动epsilon=z_candidate-z_teacher。

如果2*max(abs(epsilon))<gamma，则可以保证预测类别不变。

该条件为充分条件，不是必要条件。

平均logits MSE下降不保证所有样本满足此条件。

不宣称该关系为SNN独有理论。

R124：

L_FT=mean_n CE(mean_t(z[t,n,:]),y[n])。

使用原始任务交叉熵。

R125：

所有原权重软阈值保持固定：

d_next=d_current。

不执行额外权重阈值增长或BN手动活动稀疏更新。

R126：

R_rec=(Acc_FT-Acc_P0)/(Acc_B0-Acc_P0)。

准确率必须统一采用比例或百分比口径。

若分母非正，则R_rec为NA并说明原因。

恢复比例不得裁剪至[0,1]。

## 5. 固定父模型

只使用EXP-027已经通过真实候选准入的75-R1模型。

实际路径：

/home/hym/keyan1/experiments/EXP-026/models/75-R1.pth

SHA256：

95073e1c9b3731a792fc72d596b3fa1c96b86989b328241b00bb663020c80958

B0参考：

/home/hym/keyan1/experiments/EXP-026/models/B0.pth

SHA256：

cd6e151efe68ee77ef57ce474d0e83737321f271d4bace1ef8809013a66ae60d

以以下真实文件进行交叉核查：

experiments/EXP-026/checkpoint_manifest.json

experiments/EXP-027/fixed_model_manifest.json

experiments/EXP-027/candidate_admission_lock.json

全部父检查点保持只读。

运行前后核对SHA256。

若文件缺失或哈希不一致，停止任务并报告。

不得自动使用75-R3或其他模型替代。

## 6. 结构与代码审查

优先读取：

vendor/UniSparse/models/cifarnet.py

vendor/UniSparse/models/submodules/layers.py

vendor/UniSparse/utils/utils.py

experiments/ROUND-017/continue_train.py

experiments/ROUND-017/common.py

experiments/ROUND-025/run.py

experiments/ROUND-025/gate.py

复用已经验收的紧凑模型构建及加载代码。

从实际检查点读取state_dict及必要的结构信息。

不得虚构模型构造函数。

75-R1实际结构：

fc1输入16384，输出512。

fc2输入512，输出100。

Boost及时间平均方式不变。

物理参数量11399524。

保持所有原有脉冲神经元类型、替代梯度、池化及其他卷积结构不变。

必须验证相关模块注册引用均指向紧凑层，没有旧2048通道层的残留参数或隐式计算路径。

不得修改vendor官方源码。

## 7. 实验分组

B0：

原始UniSparse。

无新增训练。

P0：

原始75-R1紧凑模型。

无新增训练。

F0：

从75-R1独立加载。

仅训练fc2的潜在权重和bias。

冻结其他全部参数。

特征提取网络保持eval模式，BN运行统计不得更新。

fc2按实际训练需求设置模式。

原神经元状态仍需每批正确reset。

允许fc2原有软阈值参数化下的有效连接自然变化。

F1：

从完全相同的75-R1独立加载。

训练全部可训练参数。

采用正常model.train()，允许BN affine及运行统计正常适应。

保持所有原权重软阈值固定。

两个训练组均不增加手动稀疏惩罚。

不引入新STE、额外损失、蒸馏、剪枝掩码锁定或模型结构改变。

F0用于定位仅分类读出适应的可恢复性。

F1用于建立传统全模型微调强对照。

F0与F1改变的训练参数范围及BN模式不同，不能将两者差异归因于单一神经元动力学机制。

## 8. 数据隔离

沿用既有固定数据划分：

experiments/EXP-004/splits/indices.json

SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

训练集45000张。

验证集4500张。

每轮训练使用175个完整batch，每batch256张。

drop_last=True。

每轮实际训练44800张，丢弃200张。

继承EXP-018训练数据增强和归一化方式。

两个训练组使用相同的每轮采样顺序、尾批丢弃索引及增强随机种子。

正式验证使用确定性评价预处理，保留尾批。

原500张诊断池不使用。

禁止读取官方test_batch。

不使用验证集调节学习率、提前停止、选取最佳Epoch或改变训练设置。

原验证集已经多轮用于开发，不能宣称为独立最终测试集。

## 9. 固定优化配置

CifarNet，原base_channels=256。

T=4。

LIF：

tau=2。

v_threshold=1。

v_reset=0。

detach_reset=True。

ATan替代梯度。

CuPy后端。

训练batch_size=256。

workers=4。

SGD，momentum=0.9。

weight_decay=0。

初始学习率0.01。

新建5轮CosineAnnealingLR，eta_min=0，T_max=5。

每个Epoch结束后执行一次scheduler.step()。

不得每个Batch执行调度。

固定5轮实际学习率序列并在训练前记录。

训练AMP=True。

使用新建GradScaler。

accumulation_steps=1。

seed=12450。

F0/F1每轮使用相同预定采样及worker随机种子。

建议沿用原EXP-018的epoch-specific随机控制方式，在本轮使用本地Epoch1—5。

使用独立新优化器，不恢复原300轮优化器Momentum或已结束的调度器。

不得改变学习率或尝试其他配置。

## 10. 权重稀疏机制

继续使用官方有效权重函数：

W_eff=sign(W_raw)*PseudoRelu(abs(W_raw)-d)。

所有SparseConv2d.threshold固定。

本轮不调用产生非零稀疏惩罚更新的penalize路径。

不对SparseBN手动减去活动惩罚。

实际手动稀疏更新次数必须为0。

可以按需要清理原有临时惩罚状态及pre_norm_input缓存，但不得修改科学算法或持久化冻结状态。

保留原任务梯度及PseudoRelu。

不增加硬掩码。

原无效连接可以通过任务梯度自然重新激活。

记录：

有效连接新增数量。

有效连接失活数量。

逐层有效连接密度。

全部稀疏层权重稀疏率。

实际非零有效权重总数。

不得将有效连接重新激活描述为本轮新增算法。

物理模型参数始终保持11399524。

## 11. 实际训练候选准入

复用ROUND-025对象级准入思想。

通用正确性和真实候选验证必须分开。

在正式训练前完成：

1. 原父检查点及固定结构验证。
2. F0/F1初始state_dict完全一致。
3. 初始真实前向输出与75-R1一致。
4. 严格FP32、实际CuPy后端及reset检查。
5. 真正2+1不等尾批的评价统计测试。
6. F0优化器仅包含fc2.weight和fc2.bias。
7. F0在真实训练批次更新后，所有非目标参数及BN持久化buffer保持不变。
8. F1真实反向传播及SGD更新正常。
9. 两组所有软阈值更新前后完全一致。
10. 手动稀疏更新次数为0。
11. 两组CE计算与原任务一致。
12. AMP、GradScaler及非有限数值检查。
13. 紧凑模型优化器状态保存及恢复正确。
14. 两组无旧结构残留、参数量不变。
15. 实际训练样本、随机种子和资源约束正确。

所有有参数更新的预检必须在临时模型上完成。

正式训练从原只读75-R1独立初始化。

不得使用已经完成预检更新的临时模型继续正式训练。

记录真实样本、输入形状、断言、观察值及退出状态。

## 12. 机器可核验训练准入锁

建立本轮独立准入机制。

必须明确绑定：

父检查点哈希。

实际F0/F1初始化状态。

优化器参数范围。

训练配置。

固定数据划分。

实际代码SHA。

必需测试集合。

真实通过测试集合。

不得只检查测试数量。

不得硬编码admission=True。

至少加入缺失候选测试、模型哈希不一致、错误优化器范围、样本覆盖不足、测试断言失败及配置变化的负面拒绝测试。

所有负面测试必须通过。

正式训练入口必须主动验证训练准入锁。

完成后提交GitHub并核验远程锁。

未通过全部准入条件时不启动正式训练。

## 13. 训练执行

F0、F1各固定训练5轮。

每轮175个计划训练batch。

每组875次计划优化器尝试。

实际成功更新次数和AMP跳步分别记录。

正常AMP跳步不必自动判定实验无效，但必须保存。

训练每批：

- 清空梯度。
- 正确设置训练和冻结模块模式。
- 执行真实SNN前向。
- 沿时间维度平均logits。
- 计算交叉熵。
- AMP scale及反向传播。
- 执行SGD更新。
- 更新scaler。
- 正确reset神经元状态。
- 清理必要临时缓存。
- 不执行原手动稀疏惩罚。

F0建议在特征提取部分不保留反向计算图，确保冻结状态和实际训练语义正确。

F1正常端到端替代梯度训练。

固定第5轮终点为主要评价模型。

不根据验证集选择best模型。

每轮保存原子恢复检查点。

保存第5轮独立纯推理检查点。

不得自动开展第6轮。

## 14. 每轮记录

记录：

训练Accuracy及CE。

实际学习率。

实际样本数量。

采样顺序哈希。

尾批丢弃索引。

优化器成功更新次数。

AMP跳步次数。

手动稀疏更新次数。

参数有限性。

全部稀疏层阈值。

逐层有效连接密度。

新增及失活连接数量。

物理参数量。

梯度及参数更新必要统计。

训练耗时。

GPU峰值allocated/reserved。

CPU/GPU资源累计。

F0冻结参数及BN运行状态审计。

最终至少核查全部原冻结参数，而不是抽样少量参数。

## 15. 终点评价准入锁

训练结束后不能直接使用原训练前准入锁替代正式终点模型审计。

分别核查实际F0/F1第5轮检查点：

模型哈希。

结构及参数量。

阈值固定。

全部参数有限。

推理状态及reset正确。

加载前后前向一致。

训练协议、采样顺序及Epoch完整性。

F0的非目标参数及持久化buffer保护。

F1的完整训练范围。

实际成功更新及AMP记录。

验证无未授权配置变化。

使用实际最终检查点生成独立evaluation_admission_lock.json。

包括最终模型哈希、对应训练记录、必要断言和退出状态。

完成远程提交及核验后，正式评价入口再次核查实际模型身份。

缺少任一终点对象级检查时，不执行正式评价。

## 16. 正式评价

四组：

B0。

P0。

F0。

F1。

均使用固定4500张验证样本。

严格FP32。

关闭cuDNN TF32。

关闭matmul TF32。

关闭AMP。

eval模式。

Batch Size=64。

workers=4。

保持尾批。

正确reset神经元。

重新计算全部真实指标。

不直接复制历史B0/P0数据作为本轮正式结果。

核查B0/P0与有效EXP-027历史结果。

若结果不同，如实报告并核查环境、精度、模型和数据。

不得人为修改新结果匹配历史结果。

## 17. 评价指标

主要：

Accuracy。

CE。

R126准确率损失恢复比例。

实际物理参数数量。

实际结构压缩率。

SOP_sample_weighted。

SOP_author。

逐层SOP。

FR_global。

逐层FR。

有效非零权重数。

有效权重稀疏率。

相对P0新增及失活的有效连接。

纯推理模型文件大小。

训练墙钟。

GPU及CPU资源。

逐样本真实预测、标签、正确性及CE。

B0和P0的FR/SOP口径必须与EXP-027一致。

物理参数量下降、有效权重稀疏率和理论SOP严格区分。

不将理论SOP解释为硬件实测能耗或GPU实际加速。

## 18. 探索性机制诊断

在固定D_cal或已采集验证输出上，可按R123统计：

教师预测类别的原始分类间隔。

实际logits扰动的无穷范数。

满足充分条件的样本比例。

实际预测一致率。

该诊断只使用已锁定模型。

不得影响训练或候选选择。

不实现新的margin损失函数。

不根据诊断调整模型。

不能将充分条件不满足解释为预测一定改变。

如果资源不足，可标记该辅助诊断NA，不影响主要实验。

## 19. 预注册科研假设和推进门槛

H-028：

F1能够通过普通任务微调恢复具有实际意义的性能损失。

主要条件：

1. F1相对P0准确率至少提高1个百分点。
2. F1验证CE低于P0。
3. F1实际物理参数量保持11399524。
4. F1实际SOP不超过原B0。
5. 全部科学和工程检查有效。

另单独报告较严格的SOP保护：

F1_SOP<=1.01*P0_SOP。

较严格SOP保护是辅助结果，不替换原主要联合条件。

H-029：

在相同数据顺序、计划更新次数及训练轮数下，全模型适应相对仅分类读出微调具有额外价值。

主要性能条件：

F1相对F0准确率至少提高0.3个百分点。

且F1验证CE低于F0。

同时报告F0/F1实际SOP差和训练开销差。

两组训练范围及BN模式不同，因此不能解释为单一机制因果证明。

若F1未达到H-028门槛：

只能说明固定5轮、固定学习率协议下未达到要求。

不能据此宣称充分训练的标准微调普遍无效。

若F0与F1均恢复良好：

说明简单任务适应可能已经足够。

不自动提出新的复杂重构方法。

以上为预注册探索性门槛，不是统计显著性定义。

## 20. 配对统计

固定2000次配对Bootstrap。

seed=20261128。

numpy.default_rng PCG64。

百分位95%置信区间。

比较：

F0-P0。

F1-P0。

F1-F0。

F0-B0。

F1-B0。

报告：

Accuracy百分点差。

CE差。

95%配对区间。

参考错而候选对的样本数。

参考对而候选错的样本数。

预测类别不同数量。

若逐样本SOP未采集，SOP配对区间标记NA。

目前仅一个父训练种子及一个微调随机种子。

不能宣称跨训练种子稳定性。

不能将验证集样本级Bootstrap解释为独立训练重复统计。

## 21. 资源预算

需要用户新授权。

建议：

一张符合实验室共享管理规定的RTX4090。

GPU总占用上限30分钟。

CPU累计计算上限90分钟。

新增正式训练总量最多10个Epoch。

优先选择剩余显存充足的GPU。

所有前置检查、临时模型更新、正式训练、失败重试、评价及诊断计入预算。

完整记录进程树CPU计算时间及保守预算预留。

不得将GPU进程墙钟描述为GPU内核精确执行时间。

不使用第二张GPU。

不影响其他用户进程。

不因会话重启而重置累计预算。

复用已经验收的持久化进程、原子检查点及资源终止机制。

## 22. 停止条件

以下情况停止对应任务：

- 原父模型或数据哈希不一致。
- 真实结构、初始化或加载检查失败。
- 准入覆盖不完整。
- 无法完成必要远程锁。
- F0冻结参数或持久化buffer发生未授权变化。
- 原权重软阈值变化。
- 出现手动稀疏惩罚更新。
- 物理参数量变化。
- 参数、损失或实际更新非有限。
- OOM。
- 资源超限。
- 关键实验配置不一致。

不得自动降低Batch Size、改变学习率、训练轮数或AMP配置。

未完成全部5轮的组不得作为固定终点主要对照。

保留失败和部分完成检查点。

## 23. 科研状态更新

更新：

research/THEORY.md。

research/HYPOTHESES.md。

research/RESEARCH_STATE.md。

research/EXPERIMENT_PLAN.md。

research/EXPERIMENT_REGISTRY.csv。

research/DECISION_LOG.md。

新增：

research/ROUND-026-TASK.md。

追加理论版本v0.3.2及R123—R126。

新增H-028、H-029，初始状态UNTESTED。

保留原H-026/H-027限定判定。

EXP-026永久INVALID。

EXP-027保持COMPLETED/VALID。

原固定支持集读出重构方法STOPPED。

UniSparse继续保持Controlled Baseline。

EXP-028定位为传统微调强对照实验，不标记为新论文创新。

AGENTS.md追加训练准入和最终候选评价准入必须分别验证的规范。

## 24. 实验交付

新增独立目录：

experiments/ROUND-026/

experiments/EXP-028/

至少保存：

- TASK.md
- config.json
- environment.txt
- code_version.json
- commands.sh
- status.json
- parent_checkpoint_audit.json
- architecture_audit.json
- initial_model_audit.json
- optimizer_parameter_audit.json
- formula_tests.json
- preflight_tests.json
- admission_coverage_matrix.csv
- admission_negative_tests.json
- training_admission_lock.json
- training_admission_lock_commit.json
- epoch_metrics.csv
- training_sample_audit.csv
- threshold_audit.csv
- frozen_parameter_audit.json
- layer_sparsity.csv
- topology_transition.csv
- checkpoint_manifest.json
- evaluation_admission_lock.json
- evaluation_lock_commit.json
- validation_summary.csv
- validation_per_sample.csv
- layer_sops.csv
- paired_bootstrap.json
- hypothesis_gate_results.json
- resource_usage.csv
- REPORT.md
- limitations.md
- run.log
- figures/

可选机制诊断：

margin_diagnostics.csv

必须标注其探索性质。

图表至少包含：

训练CE和Accuracy曲线。

四组固定终点性能。

实际参数量及SOP。

原剪枝损失恢复比例。

逐层有效连接变化。

配对准确率差及置信区间。

未执行部分标记NA及真实原因。

大型权重保留服务器，记录实际路径、结构和SHA256。

不覆盖历史结果。

## 25. GitHub交付

从ROUND-025最终结果提交创建独立分支。

读取并遵守AGENTS.md。

正式训练前提交任务、实际代码、固定配置、预注册门槛及训练准入锁。

正式评价前提交实际终点模型准入锁。

结束后提交全部真实结果及科研状态更新。

保留唯一Run ID。

不修改main。

不force push。

不覆盖EXP-026或EXP-027。

最终返回：

- GitHub实验分支链接。
- 实际执行代码SHA。
- 训练准入锁SHA。
- 最终评价准入锁SHA。
- 最终结果提交SHA。
- EXP-028执行状态及有效性。
- F0/F1实际完成Epoch。
- 四组真实Accuracy、CE、FR、SOP及参数量。
- 两组性能恢复比例。
- 有效连接变化及阈值保护检查。
- F0冻结状态审计。
- 配对统计及H-028/H-029门槛结果。
- GPU及CPU实际资源消耗。
- 全部异常、限制和未完成项目。
- 最终模型路径及SHA256。
- 完整实验报告和原始数据路径。

本轮不提出新的论文创新，不重新剪枝，不恢复历史重构方案，不开展超参数搜索。

获得用户授权后执行。