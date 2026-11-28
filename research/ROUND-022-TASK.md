# Codex科研执行任务：ROUND-022

# UniSparse精确脉冲判别贡献与联合神经元结构化剪枝

## 1. 任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-022
- 实验编号：EXP-024
- 科研决策：DEC-022，STOP + PIVOT
- 理论版本：v0.2.8
- 科研阶段：结构化压缩理论验证与最小方法先导
- 科学问题：Q-012
- 新科研假设：H-024、H-025
- 候选方法：ESC-JP，Exact Spike-Contribution Joint Pruning
- 项目仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-021-spike-exposure-pruning
- 本轮建议分支：research/round-022-exact-spike-contribution
- 上轮实际执行SHA：a966792881e15b985dfe1bc148daed5d2223915b
- 上轮最终结果SHA：b3d35734e7a73c6b4e3aa13e40cd6c374fd9d80a
- UniSparse官方SHA：b5b3da4fcd2e7abcd2c20c93d41cf63ef3201daa
- 依赖实验：EXP-018、EXP-019、EXP-023，以及历史结构化剪枝EXP-014

优先级：

P0：归档旧候选、审计目标网络结构及精确损失公式。

P0：真实前向、代数重构和结构删除等价性。

P1：固定25%隐层剪枝预算下的公平方法比较。

本轮不开展SGD微调或完整训练，不改变原神经元动力学，不使用官方测试集。

## 2. ROUND-021正式科研判定

EXP-023保持COMPLETED，但附加训练准入协议偏差记录。

H-022更新为SUPPORTED，限定当前U2第100轮模型及D_grad/D_cal的实际暴露差异。

D_grad A_p=0.3701872750377691。

D_cal A_p=0.3706867484531585。

H-023更新为CONTRADICTED，限定固定alpha=0.25、指定父模型、101—110轮真实训练和预注册推进条件，并保留准入偏差限制。

第110轮：

W0：

- Accuracy=91.6222%。
- CE=0.258972。
- SOP/sample=90878209.928。
- 权重稀疏率=11.2487%。

W1：

- Accuracy=90.9778%。
- CE=0.261916。
- SOP/sample=91954854.41244444。
- 权重稀疏率=11.3676%。

W2：

- Accuracy=91.6667%。
- CE=0.252036。
- SOP/sample=90602112.68755555。
- 权重稀疏率=11.4149%。

W1相对两个对照的路径A、B均失败。

停止SEAWP当前固定层级惩罚分配方案。不调整alpha，不追加训练，不引入EMA、梯度保护或额外动态反馈。

MP-EFR/TP-EFR保持停止状态。

UniSparse Controlled Baseline继续保留。

## 3. 训练准入偏差的正式处理

EXP-023训练前所谓非等尾批检查实际只有1+1样本。

真正2+1检查在训练后通过。

不能将其描述为训练前全部正确性门槛已经通过。

基于以下事实，保留EXP-023作为具有局限的探索性负面证据：

- 实际暴露估计采用512样本、Batch64。
- 真实诊断不存在不等大小尾批。
- W0历史指标复现一致。
- 真实SOP重建最大相对误差约3.80e-8。
- 训练后真正非等尾批补测通过。

本轮不重复EXP-023训练。

新增准入审计要求：

每项前置检查必须保存实际测试输入、分批数量、断言内容和真实退出状态。

准入程序不得仅依据人工命名或预填True判定成功。

必要检查存在缺失、错误标记或未运行时，不进入对应正式实验阶段。

追加AGENTS.md的准入测试覆盖及机器可核验锁定要求，保留原有科研规范。

## 4. 新研究问题及理论动机

当前UniSparse U2第300轮：

- Accuracy=94.244444%。
- CE约0.175194。
- SOP/sample约86008931.65。
- 全部稀疏层权重稀疏率约14.4702%。

原UniSparse实现使用稀疏有效权重，但底层仍为普通稠密形状卷积。

本轮研究真实结构化模型压缩，而非进一步调整稀疏惩罚。

CifarNet最后隐层的fc1：

in_channels=16384。

out_channels=2048。

kernel_size=1。

fc2：

in_channels=2048。

out_channels=100。

Boost：

AvgPool1d(kernel_size=10,stride=10)。

最终类别数10。

最后隐层神经元之后没有其他非线性脉冲神经元。

上述结构需要从实际模型确认。

## 5. 固定父检查点

使用：

experiments/EXP-018/runs/EXP-018-CONTINUE-001/U2/epoch-300.pth

SHA256：

19b1312cbaae49ad95a7d001270c4357d7ea23b2584d19698b0084b3da4b094f

保持父检查点只读。

运行前后核对SHA256。

不使用best_validation代替固定300轮模型。

所有候选均从同一父模型开始。

不加载训练优化器进行新训练。

## 6. 官方结构审查

读取：

vendor/UniSparse/models/cifarnet.py

vendor/UniSparse/models/submodules/layers.py

vendor/UniSparse/models/submodules/blocks.py

vendor/UniSparse/utils/utils.py

确认真实执行路径：

activation5

maxpool5

flatten

fc1

bn1（IdentityBN）

activation_fc1（SparseLIF）

fc2

flatten

Boost

时间平均分类输出。

必须确认：

- fc1输出维度2048。
- fc2输入维度2048。
- fc2输出维度100。
- Boost为每连续10个输出取平均。
- 最后一组LIF神经元没有跨神经元耦合。
- fc1、fc2没有其他分支或消费者。
- 目标模型没有额外分类非线性。
- 实际交叉熵使用时间平均logits。

若实际结构不符合精确推导条件，停止正式实验并报告，不自行寻找其他目标层。

## 7. R101—R103精确脉冲贡献

捕获activation_fc1输出：

s.shape=[T,B,H,1,1]

T=4。

H=2048。

s必须为二值脉冲。

R101：

r[n,i]=sum_t(s[t,n,i])/T

r.shape=[B,H]

使用原有效fc2权重：

W2_eff=fc2.sparse_weight()

去除末尾1×1维度：

W2_eff.shape=[100,H]

不得使用未经软阈值处理的潜在权重代替有效权重。

R102：

A[k,i]=mean(W2_eff[10*k:10*k+10,i])

A.shape=[10,H]

类别偏置：

b[k]=mean(fc2.bias[10*k:10*k+10])

若真实bias不存在，则对应偏置为0。

R103：

z_reconstructed=r @ A.T+b

真实参考：

z_true=model(x).mean(dim=0)

在数学上两者相等，实际允许浮点运算误差。

使用FP64进行离线代数计算，以原FP32完整前向作为真实参考。

首先在真实模型小样本上验证上述关系。

同时检查分类预测及CE重建。

建议归一化误差：

norm_inf(z_reconstructed-z_true)/max(1,norm_inf(z_true)) <= 1e-5。

若超出容差，必须核查Boost排列、有效权重、偏置、脉冲捕获及神经元状态。

不得通过放宽容差隐藏公式错误。

## 8. R104—R108精确剪枝损失

每个神经元的贡献：

v[n,i,k]=r[n,i]*A[k,i]

对删除集合S：

R104：

z_minus_S=z-sum_{i in S}(v[:,i,:])

R105：

L(S)=mean_n(CE(z_minus_S[n],y[n]))

Delta_L(S)=L(S)-L(empty)

交叉熵：

CE(z,y)=logsumexp(z)-z[y]

使用数值稳定的logsumexp。

R106：

一般情况下Delta_L({i,j})不等于Delta_L({i})+Delta_L({j})。

R107：

目标是在固定删除数量m时最小化L(S)。

R108：

conditional_score(i,S)=L(S union {i})-L(S)

不计算替代梯度。

不需要任务反向传播。

不使用泰勒展开估计真实删除损失。

精确性仅针对指定结构、固定权重和保留神经元动力学不变条件。

分块贪心策略不是全局最优组合优化求解器。

## 9. 固定数据与实验环境

沿用已经通过审计的UniSparse专用环境：

CifarNet。

base_channels=256。

T=4。

CuPy LIF。

tau=2。

v_threshold=1。

v_reset=0。

detach_reset=True。

FP32真实网络推理。

Batch Size=64。

workers=4。

确定性评价预处理。

model.eval()。

所有模型前向前后按原协议reset神经元状态。

固定数据：

EXP-020原D_grad：512张，用于剪枝集合选择。

EXP-020原D_cal：512张，仅用于锁定后泛化诊断。

原4500张验证集用于最终比较。

原完整划分：

experiments/EXP-004/splits/indices.json

SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

D_grad索引哈希：

93c4c834c3fd3c8b92a64caa88da8a671e27be23dc2df11999c8e5f3dfbe47d8

D_cal索引哈希：

b671cbd803b13cc76b02f5db4bad311ef9492276622051df01af56a931301baa

必须读取并核对历史实际索引。

不重新采样。

D_grad/D_cal曾参与原模型训练，不得称为独立未见数据。

原验证集多轮用于开发，不是最终确认性评价。

旧500张诊断池不使用。

禁止读取官方test_batch。

## 10. 固定剪枝率与对照组

删除最后隐层512个神经元。

保留1536个。

固定剪枝率25%。

不搜索其他剪枝率。

所有剪枝组必须删除完全相同数量的神经元。

B0：

原始U2，不剪枝。

S0：

固定随机删除512个。

使用numpy.default_rng(20261128)，无放回选择。

保存随机生成器及实际索引。

S1：

fc1有效权重L1排序。

score[i]=sum_j(abs(W1_eff[i,j]))

不包括bias。

删除得分最小的512个。

S2：

平均脉冲发放率排序。

score[i]=mean_n(r[n,i])

删除得分最小的512个。

S3：

精确单神经元删除损失。

score[i]=L({i})-L(empty)

一次计算全部2048个神经元的精确单独损失。

删除得分最小的512个。

M5：

精确条件边际损失分块贪心。

每轮删除16个神经元，共32轮。

所有排序平局使用较小原始神经元索引优先。

保留全部真实负损失增量，不取绝对值。

不能依据D_cal或验证集结果改变任何选择。

## 11. M5算法伪代码

输入：

D_grad真实r、A、b、标签。

删除数m=512。

block_size=16。

初始化：

S=empty

z_current=r @ A.T+b

重复直到len(S)=512：

    current_loss=mean_CE(z_current,y)

    对每个尚未删除的神经元i：

        candidate_logits=z_current-r[:,i,None]*A[None,:,i]

        score[i]=mean_CE(candidate_logits,y)-current_loss

    按score升序及原始索引排序

    Q=前16个候选

    S=S union Q

    z_current=z_current-sum_{i in Q}(v[:,i,:])

    重新计算实际联合CE

    保存本轮全部选择索引、边际得分、联合CE及运行开销

输出：

S。

注意：

同一分块内的16个神经元边际得分均基于分块开始时的S。

不能将分块内边际损失之和作为真实联合损失。

每一轮必须使用实际联合logits重新计算后续候选。

每轮实际联合logits应定期与R104直接重算结果核对。

## 12. 搜索计算开销限制

不重复执行完整SNN前向计算每个候选。

只允许利用已捕获的r和A进行离线候选损失计算。

候选计算复杂度约为：

O((m/block_size)*N*H*K)

其中：

N=512。

H=2048。

K=10。

采用固定候选分块处理，避免巨大中间张量。

不使用不必要的全网络梯度。

不实现全局穷举或额外搜索方法。

记录：

- 捕获脉冲耗时。
- S3选择耗时。
- M5选择耗时。
- 峰值额外显存。
- 结构构建耗时。
- 实际完整验证前向耗时。

## 13. 精确性与组合效应验证

至少测试：

1. 全零删除集合恢复原logits。
2. 单神经元删除预测。
3. 多神经元联合删除预测。
4. 实际零脉冲门控结果。
5. 真正结构化删除结果。
6. 保留神经元脉冲逐元素一致。
7. 输出分类预测一致。
8. 交叉熵重建一致。
9. 不等大小尾批的样本加权。
10. 捕获功能开关不改变模型输出。

在预先固定的真实32张D_grad样本上，选择固定单神经元、16神经元及512神经元删除集合进行检查。

不得依据实际性能寻找容易通过的测试集合。

组合效应诊断：

比较S3和M5最终删除集合：

实际联合Delta_L(S)。

独立单神经元Delta_L({i})之和。

两者差异。

报告静态排序与联合损失选择之间的差异。

数学上存在非加性不等于它具有实际性能价值。

## 14. 真实结构化删除

所有候选均构建真实紧凑模型。

不能只在完整模型输出上乘以mask后就宣称参数量减少。

设保留神经元索引R，长度1536。

保持原始索引升序。

fc1：

新输出通道数1536。

新原始权重=parent.fc1.weight[R,:,:,:]。

新bias=parent.fc1.bias[R]。

保留原软阈值buffer。

fc2：

新输入通道数1536。

新原始权重=parent.fc2.weight[:,R,:,:]。

保留原bias及软阈值buffer。

activation_fc1：

保留原LIF类型、参数及重置方式。

所有其余卷积、BN、池化及Boost保持不变。

检查模块注册引用是否仍指向真实紧凑层。

不得残留指向原2048通道模块的隐式计算路径或参数引用。

父检查点保持只读。

实际紧凑模型应与同集合零脉冲门控参考在数值容差内等价。

若涉及替换模块，优先在独立研究代码中实现，不修改vendor官方源码。

## 15. 物理参数数量审计

每个删除神经元理论删除：

fc1权重16384个。

fc1偏置1个。

fc2权重100个。

合计16485个。

删除512个：

8440320个参数。

根据官方源码推导原模型总参数预计36720484。

紧凑模型预计28280164。

上述总参数以真实模型named_parameters独立计算为准。

若存在差异，解释实际参数结构。

分别报告：

- 实际参数张量元素总数。
- 有效非零权重数量。
- 有效权重稀疏率。
- 稠密FP32参数载荷估计。
- 实际保存的纯推理模型文件大小。
- 检查点格式和是否包含额外buffer。

不将含优化器状态的训练检查点大小与纯推理模型直接比较。

不将原模型非结构化有效稀疏率与实际结构删除比例混为一谈。

## 16. 选择锁与评价顺序

首先完成：

D_grad上所有剪枝集合选择。

保存S0/S1/S2/S3/M5全部删除与保留索引。

保存完整r、A及标签的必要哈希。

保存实际代码SHA及所有固定配置。

在D_cal或4500张验证集评价前提交selection_lock.json。

锁定之后：

不得更换排序准则、block_size、随机种子、删除比例或神经元集合。

D_cal仅用于锁定后观察，不能据此调参。

完成全部必要正确性检查后，评价所有真实紧凑模型。

## 17. SOPs监控

保留EXP-018/019的原7层理论SOP监控口径。

对紧凑模型实际消费者输入和有效掩码重新计数。

特别记录：

- fc1 SOP。
- fc2 SOP。
- 两层合计SOP。
- 全网络SOP_sample_weighted。
- SOP_author。

实际输入必须为二值脉冲。

保留验证尾批，按实际样本总数加权。

R93—R95仍可用于固定输入下的直接成本核验。

若采用新增逐样本SOP统计，必须与原聚合监控器进行独立校验。

不将理论SOPs解释为GPU实际运行时间或硬件实测能耗。

## 18. 正式评价指标

4500张固定验证样本。

报告各组：

Accuracy。

CE。

FR_global。

逐层FR。

全网络SOP/sample。

逐层SOP。

实际参数总量。

有效非零权重数。

结构化压缩率。

有效权重稀疏率。

选择耗时及峰值显存。

真实模型文件大小。

逐样本预测、标签、正确性和CE。

如果条件允许，可以在相同硬件、相同Batch、预热及CUDA同步条件下提供探索性推理耗时对比。

该耗时测量不属于本轮强制推进条件。

共享GPU负载下的时间不能直接代表普适实际加速。

## 19. 预注册推进门槛

M5进入下一轮统一微调验证的必要条件：

1. 所有精确性与结构审计通过。
2. 真实物理参数总量下降至少20%。
3. M5相对B0准确率下降不超过0.5个百分点。
4. M5相对S1准确率提高至少0.3个百分点。
5. M5相对S2准确率提高至少0.3个百分点。
6. M5相对S3准确率提高至少0.3个百分点。
7. M5全网络SOP不超过B0的101%。
8. 所有剪枝组物理参数预算相同。
9. 全部样本配对及数据隔离正确。

以上为探索性资源推进条件，不是统计显著性判定。

S0随机对照结果也必须完整报告。

若M5优于L1和FR，但未优于S3：

不能宣称条件边际损失重估具有独立价值。

若所有剪枝方法均保持较高准确率，但M5没有独立优势：

说明该层可能存在可利用的结构冗余，但本候选剪枝准则尚不能作为论文核心创新。

若M5未通过：

不自动改变剪枝比例或搜索其他block_size。

返回真实结果，由理论研究员决定是否进入一次公平微调验证或终止。

## 20. 配对统计

固定2000次配对Bootstrap。

seed=20261128。

numpy.default_rng PCG64。

百分位95%区间。

主要比较：

M5-S1。

M5-S2。

M5-S3。

次要比较：

M5-B0。

M5-S0。

报告：

- Accuracy百分点差。
- 平均CE差。
- 配对Bootstrap区间。
- 参考组错、候选组对的样本数。
- 参考组对、候选组错的样本数。
- 预测类别不同数量。

逐样本SOP缺失时标记NA，不生成虚构区间。

单种子、重复开发验证集结果不代表独立统计确认。

## 21. 科研假设状态

H-024：UNTESTED。

预测R104可以准确重建任意固定删除集合的真实输出。

若真实零脉冲门控或紧凑模型不能通过归一化误差门槛，则需先检查推导条件及工程实现。

H-025：UNTESTED。

预测条件边际损失重估有助于提高相同结构压缩预算下的任务性能。

其性能判断使用第19节预先设定的条件。

公式正确不代表H-025成立。

Codex仅报告门槛结果，最终科学状态由理论研究员判定。

## 22. 文献及创新性限制

追加LITERATURE.md：

Li等，Towards Efficient Deep Spiking Neural Networks Construction with Spiking Activity Based Pruning，ICML 2024。

https://proceedings.mlr.press/v235/li24bz.html

Molchanov等，Importance Estimation for Neural Network Pruning，CVPR 2019。

https://research.nvidia.com/publication/2019-06_importance-estimation-neural-network-pruning

Luo等，ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression，ICCV 2017。

https://arxiv.org/abs/1707.06342

Good Subnetworks Provably Exist: Pruning via Greedy Forward Selection，ICML 2020。

https://proceedings.mlr.press/v119/ye20b.html

不得宣称贪心选择、线性读出分解或直接任务损失评价本身为学术首创。

当前方法也适用于具有相应线性读出的ANN，SNN特异性仍需要加强。

仅验证当前SNN结构化剪枝中的实际性能价值。

## 23. 资源预算与停止条件

需要用户新授权。

建议：

单张符合实验室共享管理规定的RTX4090。

GPU总占用上限20分钟。

CPU累计计算上限45分钟。

新增训练Epoch=0。

禁止额外剪枝率或超参数搜索。

所有检查、真实前向、候选选择、紧凑模型构建及评价均计入预算。

如果发生：

- 父模型哈希不一致。
- 数据隔离失败。
- 精确重构公式不成立。
- 紧凑模型与门控模型不一致。
- 实际物理结构没有缩小。
- 数值非有限。
- OOM。
- 超资源预算。
- 未完成必要准入检查。

停止对应正式实验并报告。

不得干扰其他用户GPU进程。

不因会话重启重置累计预算。

## 24. 科研状态及代码维护

新增：

research/ROUND-022-TASK.md

更新：

THEORY.md，追加v0.2.8及R101—R108。

HYPOTHESES.md，正式记录H-022/H-023，新增H-024/H-025。

RESEARCH_STATE.md。

EXPERIMENT_PLAN.md。

EXPERIMENT_REGISTRY.csv。

DECISION_LOG.md，追加DEC-022。

LITERATURE.md。

AGENTS.md，补充可核验的训练与实验准入锁要求。

代码新增：

experiments/ROUND-022/

实验记录：

experiments/EXP-024/

不覆盖原EXP-023。

不修改UniSparse原始vendor源码。

不删除历史负面结果。

## 25. 实验交付

至少保存：

- TASK.md
- config.json
- environment.txt
- code_version.json
- commands.sh
- status.json
- parent_checkpoint_audit.json
- architecture_audit.json
- exact_readout_tests.json
- admission_gate_audit.json
- structural_equivalence_tests.json
- data_split_audit.json
- neuron_statistics.csv
- single_neuron_loss.csv
- greedy_selection_trace.csv
- selected_neurons.json
- selection_lock.json
- interaction_diagnostics.csv
- calibration_summary.csv
- validation_summary.csv
- validation_per_sample.csv
- layer_sops.csv
- parameter_count_audit.json
- checkpoint_manifest.json
- paired_bootstrap.json
- hypothesis_gate_results.json
- resource_usage.csv
- REPORT.md
- limitations.md
- run.log
- figures/

图表至少包括：

1. 神经元发放率与真实单神经元删除损失关系。
2. 静态独立损失与真实联合损失差异。
3. M5分块贪心选择过程。
4. 各组真实参数数量及SOP。
5. 各组验证Accuracy及CE。
6. 主要配对比较及置信区间。

未执行部分标记NA及真实原因。

保留全部失败记录、重试和独立Run ID。

## 26. GitHub交付

从上轮最终结果SHA创建独立研究分支。

正式运行前提交任务、固定配置、公式、门槛及代码。

评价前提交神经元集合及selection_lock。

完成实验后同步全部真实数据及报告。

不合并main，不force push。

模型权重保留服务器，记录真实路径、模型结构配置及SHA256。

最终返回：

- GitHub实验分支链接。
- 实际执行代码SHA。
- 选择锁SHA。
- 最终结果提交SHA。
- EXP-024执行状态。
- H-024精确重构结果。
- 门控与真实紧凑结构等价性。
- 各组Accuracy、CE、FR、SOP。
- 实际物理参数量及压缩率。
- 单神经元损失与联合损失差异。
- M5相对S1/S2/S3配对结果及区间。
- 全部推进门槛结果。
- 真实资源消耗。
- 异常、偏差及实验局限。
- 报告及原始数据路径。

本轮不启动SGD微调或正式训练。

获得用户授权后执行。