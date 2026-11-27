# Codex 科研执行任务：ROUND-016

# UniSparse 受控训练工作点筛选与正式复现准备

## 1. 任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-016
- 科研阶段：新Baseline阶段0，真实训练先导及复现准备
- 科研决策：DEC-016，CONTINUE + BASELINE
- 理论版本：v0.2.2
- 科学问题：Q-010，权重稀疏性与脉冲活动的耦合优化
- 实验编号：EXP-017
- 项目仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-015-unisparse-admission
- 本轮建议分支：research/round-016-unisparse-pilot
- 上轮最终结果SHA：e1fa5ab44cc955db6d7cd5069101bb5b067ca11c
- UniSparse官方固定SHA：b5b3da4fcd2e7abcd2c20c93d41cf63ef3201daa
- 依赖实验：EXP-015、EXP-016
- 优先级：P0训练配置与lambda锁定；P1三组真实训练先导；P2训练工作点及资源评估

首先读取AGENTS.md、RESEARCH_STATE.md、THEORY.md、HYPOTHESES.md、UNISPARSE_BASELINE.md、ROUND-015-TASK.md以及EXP-016报告、环境记录和资源数据。

从上轮最终结果提交建立独立研究分支。

本轮不修改原始UniSparse科学算法，不实现ECAS，不运行完整300轮训练。

## 2. ROUND-015正式科研判定

EXP-016为COMPLETED。

UniSparse达到ENGINEERING_ADMITTED，仅限本轮实际工程测试条件。

正式完整训练仍为PENDING。

已确认：

- CuPy 13.5.1安装成功。
- 官方CuPy LIF前向、反向及Torch参考对照通过。
- FP32最大归一化梯度误差2.65e-8。
- FP16最大归一化梯度误差3.17e-4。
- 两种精度脉冲不一致数均为0。
- 完整CifarNet参数量36,720,484。
- Batch256、T4完整前向和反向成功。
- 7个卷积阈值及6个BN affine模块实际发生稀疏惩罚更新。
- 峰值allocated约11.84 GiB。
- 峰值reserved约13.74 GiB。
- 零惩罚平均批次耗时0.161231秒。
- 合成非零惩罚平均批次耗时0.231183秒。
- 以上测速各仅3个正式计时批次。
- 未确认正式论文非零lambda。
- 未运行完整训练。
- 未使用官方测试集。

保留第一次CuPy后端运行失败及np.int兼容补丁记录。

旧环境入口文件哈希核对通过，但原补丁目标文件修改前哈希缺失。不得将历史审计局限改写为完整环境逐文件一致性证明。

不重复已经通过的全部单元测试，只执行必要回归和训练入口检查。

## 3. 理论研究方向修正

论文PDF第7页、公式(28)—(32)相关正文已经明确说明：作者使用初始网络结构常数近似动态连接映射，以缓解潜在反馈问题。

因此：

- 不将名义扇出与实时有效扇出的差异称为作者未认识到的缺陷。
- 不宣称直接使用有效连接扇出具有原创性。
- 不直接实现ECAS。
- 保留未来研究动态连接感知惩罚及优化稳定性的可能性。

H-016：

PARTIALLY_SUPPORTED，或按AGENTS.md允许的枚举进行明确映射。

固定掩码、固定活动条件下的数学及合成机制已获得支持；正常训练后网络中的失配幅度及任务影响仍未验证。

H-017保持UNTESTED。

## 4. 新增理论记录

### R62—R63：简化联合优化动力学

令p为连续归一化连接密度，r为连续归一化活动率。

C(p,r)=p*r

在忽略任务损失、边界、真实离散掩码与网络参数化，采用相同连续更新尺度的简化模型中：

dp/dt=-lambda*r

dr/dt=-lambda*p

delta=p-r

d(delta)/dt=lambda*delta

delta(t)=delta(0)*exp(lambda*t)

该结论仅适用于上述简化动力学及其内部区域。

不能将其当作真实UniSparse训练系统的稳定性定理。

### R64：候选相对扇出归一化

d[c]=sum(M[:,c,:,:])

d_mean=mean_c(d[c])

d_hat[c]=d0*d[c]/d_mean

适用条件为d_mean>0、直接卷积消费关系及相应近似假设。

该公式保持通道平均惩罚系数，不保证总惩罚、实际梯度范数或任务性能不变。

仅记录理论候选，不实现训练。

### R65—R68：lambda先导数量级估计

实际代码：

SparseConv2d.penalize:

threshold += lmbda * penalty

main.py传入lmbda_weight * 当前Epoch学习率。

在accumulation_steps=1及固定形状惩罚K_l条件下：

R65：

d[l,k+1]=d[l,k]+eta[k]*lambda_weight*K[l]

R66：

d[l,K]=d[l,0]+lambda_weight*K[l]*sum_k(eta[k])

R67：

q[l]=median(abs(W_initial[l]))

lambda_ref_layer[l]=q[l]/(K[l]*S300)

其中：

S300为完整300轮训练的累计学习率，即所有预期稀疏更新步骤实际学习率之和。

R68：

lambda_ref=median_l(lambda_ref_layer[l])

该方法仅用于确定原始UniSparse合理的超参数数量级。

不是新剪枝算法，不保证指定最终稀疏率，也不作为独立学术创新。

## 5. 固定源码及环境

使用：

vendor/UniSparse/

保持官方源码不变。

使用ROUND-015验收通过的UniSparse专用虚拟环境。

实际环境路径必须先核对，预期为：

.local/round014-venv

核查：

- Python 3.11.9。
- PyTorch 2.4.1+cu118。
- torchvision 0.19.1+cu118。
- CuPy 13.5.1。
- SpikingJelly 0.0.0.0.14及已记录的局部兼容补丁。
- NumPy 1.26.4。
- timm 1.0.22。

不修改旧AT-LIF环境，不重新安装CUDA或驱动，不升级核心依赖。

记录实际Python解释器、依赖版本、模块路径及补丁哈希。

## 6. 原论文非零lambda确认

在GPU训练前，有限度检查：

- 正式论文正文及附录。
- 官方README。
- 官方配置文件。
- 官方公开提交历史或明确的作者说明。
- 可核实的补充材料。

建议额外检索和静态核查墙钟上限10分钟，不因反复查找lambda而无限延迟研究。

不得编造作者配置。

若确认了对应CifarNet的完整非零lambda配置，保存：

- 实际数值。
- 来源链接。
- 对应模型和数据集。
- 公式或表格位置。
- 配置版本或提交SHA。

若仍无法确认，记录UNKNOWN，进入R67—R68的受控数量级初始化。

## 7. 训练数据协议

继续使用：

experiments/EXP-004/splits/indices.json

SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

训练集45000张。

验证集4500张。

原500张科研诊断池继续隔离。

只读取CIFAR-10官方训练来源文件data_batch_1至data_batch_5。

使用已实现的：

experiments/ROUND-014/controlled_data.py

先核查实际数据增强与作者main.py一致，包括：

- RandomCrop，padding=4。
- RandomHorizontalFlip。
- CIFAR10Policy。
- ToTensor。
- 作者归一化参数。
- Cutout，1个hole，length=16。

验证使用确定性预处理。

禁止构造、读取或评价官方test_batch。

训练时保持RandomSampler或等价的可追溯随机采样。

drop_last=True。

每轮175个Batch，每轮200个样本因尾批丢弃而未使用，记录实际采样情况。

各组采用相同初始化、数据顺序及随机增强控制。

## 8. 固定模型及训练配置

- model=cifarnet。
- base_channels=256。
- num_classes=10。
- T=4。
- 官方CuPy LIF。
- tau=2。
- v_threshold=1。
- v_reset=0。
- detach_reset=True。
- ATan替代梯度。
- Batch Size=256。
- num_workers=4，作为共享资源受控设置。
- seed=12450。
- SGD。
- lr=0.1。
- momentum=0.9。
- weight_decay=0。
- AMP=True。
- accumulation_steps=1。
- TET=False。
- label_smoothing=0。
- Mixup及CutMix关闭。
- Cosine学习率调度。
- num_epochs=300。
- min_lr=0。
- warmup_epochs=0。
- cooldown_epochs=0。

核对实际官方create_optimizer_v2及create_scheduler_v2行为，保存真实解析配置与参数组。

三组本轮各执行20个Epoch。

注意：scheduler完整周期必须是300轮，不能改为20轮。

本轮使用第20轮作为固定主要评价点，不依据验证集选择最有利的Epoch作为先导结果。

任务损失：

z=mean_t(model_output)

loss=CrossEntropy(z,target)

核查与官方CriterionWarpper在TET关闭时一致。

## 9. R68实际计算和配置锁定

所有候选从完全相同的随机初始化模型出发。

在不改变初始化状态的条件下：

1. 记录每层实际初始权重张量。
2. 记录权重绝对值中位数及其他必要分位数。
3. 使用真实calc_penalty逻辑确认每层形状惩罚K_l。
4. 核查flatten特殊设置及多后继连接情况。
5. 确认K_l是否在固定训练配置下保持一致。
6. 只纳入K_l>0、q_l>0且二者有限的实际被惩罚卷积层。
7. 不纳入没有权重稀疏惩罚的层。
8. 依据实际300轮scheduler逐轮计算学习率。
9. 计算S300。
10. 计算各层lambda_ref_layer及全局lambda_ref。

S300必须使用真实计划学习率和每轮175个更新机会。

官方AMP跳步后手动惩罚仍可能执行，因此不能未经说明按成功optimizer步数代替手动惩罚更新次数。

S300属于预期完整协议累计量。

不得将torch浮点阈值实际累计误差当作R66精确算术公式错误。

保存lambda_reference.csv及lambda_derivation.json。

额外记录：

- 预计第1轮累计阈值。
- 预计第20轮累计阈值。
- 预计第300轮累计阈值。
- 预计阈值相对初始权重中位数的比例。

这些是计算预测，不是实际未来稀疏率。

如果发现K_l不能按当前协议视为固定常数，暂停使用R68，报告公式适用条件不满足，不自行修改理论公式。

## 10. 实验分组

U0：

lambda_weight=0。

lambda_neuron=0。

原始UniSparse网络的零稀疏惩罚对照。

U1：

lambda_weight=lambda_ref。

lambda_neuron=lambda_ref。

U2：

lambda_weight=3*lambda_ref。

lambda_neuron=3*lambda_ref。

若第6节核实到真实论文非零配置，则允许用该配置替换U2。

除此之外，不新增候选。

若U1/U2配置意外重复，不重复执行相同组，记录原因。

必须在正式训练开始前锁定所有实际lambda、随机种子、初始化哈希、数据配置和判定规则，并提交GitHub。

禁止根据中间验证结果改变正在执行的lambda。

## 11. 保留作者原始更新机制

正式先导训练使用原始：

- SparseConv2d。
- SparseBN。
- SparseNeuron。
- calc_penalty。
- penalize。
- clear_penalty。
- 官方CuPy LIF。

保持官方训练顺序：

1. 任务前向。
2. 任务反向。
3. functional.reset_net(model)。
4. scaler.step(optimizer)或optimizer.step()。
5. scaler.update()。
6. optimizer.zero_grad()。
7. 原calc_penalty(model)。
8. 原penalize(model,lambda_weight*lr,lambda_neuron*lr)。
9. clear_penalty(model)。

U0按照作者零惩罚路径执行，不额外调用稀疏惩罚。

保持作者AMP跳步后仍执行手动稀疏更新的行为，不在本轮擅自修正。

记录AMP成功步数、跳步数、手动惩罚实际执行次数及非有限情况。

不增加：

- 动态有效扇出惩罚。
- R64归一化扇出。
- SOP预算损失。
- 知识蒸馏。
- 新替代梯度。
- 自适应lambda。
- 额外剪枝机制。
- BN重新校准。

## 12. 正式训练前置检查

必须在正式训练之前完成：

1. 官方源码固定版本检查。
2. 独立环境及CuPy必要回归。
3. 固定数据哈希检查。
4. 完整模型输出形状检查。
5. 官方任务损失及优化器参数组检查。
6. 三组初始化张量一致性检查。
7. 初始权重阈值与有效掩码检查。
8. 300轮学习率调度及恢复状态测试。
9. R65实际单步阈值更新检查。
10. U1/U2的真实有限稀疏更新检查。
11. eval状态隔离检查。
12. 受控训练入口不会访问官方测试集。
13. 检查点保存及恢复一致性检查。

前置试运行使用独立临时状态。

正式训练必须重新从锁定的相同初始化开始。

前置检查禁止改变正式初始化、优化器动量、scheduler位置或随机状态。

## 13. 正式训练与记录

三组分别完成20个Epoch。

每轮记录：

- 训练损失。
- 训练准确率。
- 验证损失。
- 验证准确率。
- 当前学习率。
- 权重稀疏率。
- 原作者FR指标。
- 样本及神经元元素加权FR。
- SOP_author。
- SOP_sample_weighted。
- 每层权重阈值。
- 每层有效连接密度。
- BN gamma/beta统计。
- AMP跳步及成功更新数。
- 实际稀疏更新次数。
- 训练、验证及保存时间。
- GPU峰值显存。
- CPU资源占用。

权重稀疏率至少分别报告：

1. 全部SparseConv2d有效权重稀疏率。
2. 实际接受权重惩罚的卷积层整体稀疏率。
3. 各层独立稀疏率。

按权重数量加权统计整体稀疏率，不能简单平均各层百分比。

未被作者惩罚的权重层单独标记。

## 14. 有效扇出诊断

只增加只读测量，不改变训练目标。

在第0、1、5、10、20轮检查点，记录各卷积层实际有效掩码。

对具有直接卷积消费关系的神经元模块，计算：

d[c]=sum(M[:,c,:,:])

记录：

- 名义扇出。
- 实际通道有效扇出。
- 平均有效扇出。
- 有效扇出的标准差。
- 变异系数。
- 最小值和最大值。
- P10、P50、P90分位数。
- 零有效扇出通道比例。

对于均值为0导致变异系数未定义的情况，标记NA并说明原因。

优先检查真实依赖映射中：

static_conv.activation -> conv1.conv

conv1.activation -> conv2

conv3.activation -> conv4.conv

conv4.activation -> conv5

具体路径以真实模型审查结果为准。

对存在MaxPooling、flatten及IdentityBN的路径单独标记适用边界，不将R59近似当成精确局部代价。

本轮不进行新的动态扇出反向传播实验。

诊断目的是了解真实训练过程中连接分布及名义扇出失配是否开始形成。

不能由20轮结果直接推断最终300轮稀疏模型的分布。

## 15. SOP统计协议

同时保存：

SOP_author：

按原作者实际聚合方式统计。

SOP_sample_weighted：

所有验证Batch的SOP总数除以实际验证样本总数。

4500张验证集使用drop_last=False。

不得丢弃验证尾批。

原Monitor跳过static_conv及其他net.skip范围，必须明确记录。

验证实际被监控输入是否为二值脉冲。

如果输入为非二值，原Monitor的直接求和只能按其定义解释，不自动称为严格二值事件计数。

报告理论SOPs，不宣称硬件实测能耗或实际推理加速。

参数总量保持原模型36,720,484，实际数量以运行审计为准。

非零权重数减少不等于稠密张量结构缩小。

## 16. 先导实验评价及筛选规则

本轮为探索性Baseline训练工作点筛选，不是新方法性能确认实验。

固定第20轮作为主要比较点。

U0为相同训练预算的零惩罚对照。

候选U1/U2满足以下全部条件，才建议进入后续完整训练审批：

1. 实际被惩罚卷积层整体权重稀疏率至少1%。
2. SOP_sample_weighted相对U0至少下降5%。
3. 验证准确率相对U0下降不超过5个百分点。
4. 每个实际被惩罚卷积层有效连接密度至少1%。
5. 损失、参数和梯度保持有限。
6. 完整20轮训练及检查点正确性审查通过。

这些是预先约定的工程筛选条件，不是统计显著性判据。

若两组均通过，优先推荐SOP_sample_weighted较低者；若相同则推荐准确率较高者；仍相同则优先U1。

若仅一组通过，推荐该组。

若两组均失败，不自动增加候选或延长训练，不事后放宽门槛。

如某候选因预算或运行错误只完成部分Epoch，不将其直接与完整20轮模型进行主要公平比较。

所有验证结果属于同一训练种子下的探索性数据。

选择lambda使用验证信息时必须如实记录；未来独立确认须采用新的训练种子及经批准的最终评价协议。

禁止将当前验证集结果称为独立最终测试性能。

## 17. 检查点与后续恢复

保存每组：

- 初始状态哈希。
- 每轮可恢复最新检查点。
- 第20轮固定检查点。
- 训练过程中验证准确率最佳检查点。
- 第5、10轮机制诊断检查点，若磁盘允许。

检查点必须保存：

- 模型参数。
- BN运行统计。
- 权重阈值及稀疏buffer。
- optimizer状态。
- scheduler状态。
- AMP scaler状态。
- 完成Epoch。
- 数据采样相关状态。
- Python/NumPy/Torch/CUDA RNG状态。
- 完整训练配置。
- 代码版本。

保存恢复一致性测试。

如果不能保证跨中断恢复后的随机增强或采样逐位一致，明确记录限制。

后续完整训练应能够从第21轮继续，沿用原300轮scheduler。

本轮不允许自动开始第21轮。

大型权重留在服务器，记录实际路径、大小及SHA256。

不上传大型权重到GitHub。

## 18. 资源与停止条件

需要用户新授权。

建议：

- 一张符合实验室共享规定的RTX4090。
- GPU总占用上限75分钟。
- CPU累计计算上限180分钟。
- 三组各20轮，总60轮。
- 优先选择实际可用显存至少16 GiB的GPU。

根据EXP-016峰值显存与真实共享负载重新检查资源。

使用可靠进程管理、心跳监控、预算截止和原子检查点保存。

启动前确保拥有足够磁盘空间。

不得干扰其他GPU用户。

CPU计算预算应尽可能覆盖DataLoader子进程，不能将墙钟时间直接等同于累计CPU时间。

发生以下情况停止受影响运行：

- 数据或版本校验失败。
- 原始公式或更新语义不一致。
- 非有限任务损失、参数或非有限手动更新。
- OOM。
- 实际资源超限。
- 关键卷积层有效连接完全消失并持续出现，按预设异常条件记录。
- 训练进程异常退出。
- 检查点无法可靠保存。

正常AMP跳步按照作者机制记录，不与参数非有限状态混为一谈。

早期准确率偏低或稀疏率未达到目标本身不构成立即停止依据，除非出现预注册的实际训练失效情形。

不根据表现擅自调整lambda、学习率、Batch Size或其他参数。

## 19. 科研状态更新

新增：

research/ROUND-016-TASK.md

更新：

- THEORY.md：追加v0.2.2及R62—R68。
- HYPOTHESES.md：H-016限定支持，H-017保持UNTESTED。
- RESEARCH_STATE.md。
- UNISPARSE_BASELINE.md。
- EXPERIMENT_PLAN.md。
- EXPERIMENT_REGISTRY.csv。
- DECISION_LOG.md：追加DEC-016。
- LITERATURE.md：记录原论文明确说明初始结构近似。

EXP-015保持PARTIAL。

EXP-016保持COMPLETED。

UniSparse工程状态保持ENGINEERING_ADMITTED。

EXP-017按真实结果登记。

在尚未完成可靠的完整训练和相应评价之前，不标记UniSparse已成功复现。

不创建新的已验证性能创新假设。

## 20. 实验交付

建立experiments/EXP-017目录及唯一Run ID。

至少保存：

- TASK.md
- config.json
- environment.txt
- code_version.json
- commands.sh
- status.json
- lambda_source_audit.md
- lambda_reference.csv
- lambda_derivation.json
- learning_rate_schedule.csv
- initialization_manifest.json
- selection_lock.json
- preflight_tests.json
- epoch_metrics.csv
- layer_sparsity.csv
- threshold_trajectory.csv
- bn_statistics.csv
- effective_fanout_statistics.csv
- validation_batch_sops.csv
- validation_per_sample.csv
- resource_usage.csv
- checkpoint_manifest.json
- summary.csv
- pilot_selection.json
- REPORT.md
- limitations.md
- run.log
- figures/

每个Run保留对应日志和独立状态。

至少绘制：

1. 三组训练和验证准确率曲线。
2. 三组训练和验证损失曲线。
3. 权重稀疏率变化。
4. 脉冲发放率变化。
5. 理论SOPs变化。
6. 各层权重阈值及有效连接密度。
7. 名义扇出与实际有效扇出分布。
8. 第20轮准确率—SOPs权衡。

失败及部分完成实验保留真实记录。

未产生的数据标记NA并说明原因，不生成虚构结果。

## 21. GitHub交付及最终汇报

正式运行前提交任务、配置、数据协议、实际lambda及执行代码。

训练完成后提交全部真实结果、异常记录和科研状态。

不修改main，不force push，不覆盖原始源码和历史实验。

最终返回：

- 实验分支链接。
- 实际执行代码SHA。
- 最终结果提交SHA。
- EXP-017执行状态。
- 原论文非零lambda是否确认。
- R68推导得到的lambda_ref。
- U0/U1/U2实际lambda。
- 各组实际完成Epoch。
- 第20轮真实验证准确率、损失、FR、SOPs及权重稀疏率。
- 逐层有效连接密度和阈值轨迹。
- 名义与有效扇出分布。
- AMP跳步及手动更新记录。
- 工作点筛选结果。
- 推荐后续完整训练的组别及原因。
- 最新检查点路径、SHA256及恢复验证结果。
- 真实GPU、CPU资源消耗。
- 更新后的完整300轮训练成本估计。
- 全部异常与实验局限。
- 实验报告和原始数据路径。

本轮目标是获得一个真实、可恢复、具有合理稀疏训练状态的UniSparse候选工作点。

不是证明任何新方法已经有效。

获得用户授权后执行本轮固定实验，不擅自扩大研究范围。