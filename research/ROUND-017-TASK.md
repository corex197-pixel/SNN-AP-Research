# Codex 科研执行任务：ROUND-017

# UniSparse Controlled Baseline完整续训与连接—活动成本分解

## 1. 任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-017
- 科研阶段：新Baseline阶段0—1，完整受控训练及理论机制诊断
- 科研决策：DEC-017，CONTINUE + BASELINE
- 理论版本：v0.2.3
- 科学问题：Q-010，连接稀疏性与脉冲活动的耦合优化
- 科研假设：H-016、H-017、新增H-018
- EXP-018：U0/U2完整受控续训
- EXP-019：连接—活动反事实成本分解
- 项目仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-016-unisparse-pilot
- 本轮建议分支：research/round-017-unisparse-baseline
- 上轮执行代码SHA：c718afecce6c65bcccbe8ba2b3e23ba07b2a7294
- 上轮结果SHA：f6f7de5cd8685dc98ca456ad0cc47929fa58691a
- UniSparse官方固定SHA：b5b3da4fcd2e7abcd2c20c93d41cf63ef3201daa
- 依赖实验：EXP-015、EXP-016、EXP-017

优先级：

P0：原始检查点、恢复训练及原始算法一致性检查。

P1：第20轮既有模型的成本分解。

P1：U0/U2分阶段续训。

P2：完整训练后理论机制诊断及科研交付。

首先读取AGENTS.md、RESEARCH_STATE.md、THEORY.md、HYPOTHESES.md、UNISPARSE_BASELINE.md、ROUND-016-TASK.md及EXP-017报告和配置。

从上轮最终结果SHA建立独立研究分支。

本轮不实现ECAS、不修改原UniSparse数学机制、不运行额外超参数搜索、不使用官方测试集。

## 2. ROUND-016正式科研判定

EXP-017为COMPLETED。

三组均完成20轮训练。

固定第20轮结果：

U0：

- Accuracy=0.832。
- CE=0.4973804389。
- FR=0.0358797936。
- SOP_sample_weighted=273551992.576。
- 受惩罚权重稀疏率约0。

U1：

- Accuracy=0.8213333333。
- CE=0.5243102536。
- FR=0.0365888458。
- SOP_sample_weighted=229828496.9867。
- 受惩罚权重稀疏率=0.0168239097。

U2：

- Accuracy=0.8162222222。
- CE=0.5439658152。
- FR=0.0373420849。
- SOP_sample_weighted=157747959.2213。
- 受惩罚权重稀疏率=0.0450717047。
- 最低受惩罚层有效连接密度=0.4044681125。

U2相对U0：

- 准确率下降1.5778个百分点。
- SOPs下降约42.333%。
- 全局FR相对增加约4.08%。

U1和U2均通过预注册先导筛选。

按预注册工作点选择规则，推荐U2。

U0/U1/U2各完成3500次成功optimizer更新，AMP跳步均为0。

U1/U2各执行3500次手动稀疏更新。

原论文CifarNet非零lambda仍为UNKNOWN。

本项目属于作者算法的Controlled Baseline，不得表述为精确复现论文表格结果。

## 3. 已确认的理论与实验观察

U2第20轮有效连接密度：

- conv1.conv：0.4044681125。
- conv2：0.4151713053。
- conv3.conv：0.4407348633。
- conv4.conv：0.8218892415。
- conv5：0.8202039931。
- fc1：0.9875636399。
- fc2：0.9998144531。

实际形状惩罚：

K_l：

- static_conv.conv=0。
- conv1.conv=1024。
- conv2=1024。
- conv3.conv=1024。
- conv4.conv=256。
- conv5=256。
- fc1=16。
- fc2=1。

第20轮U2直接消费路径的有效扇出：

static_conv.activation -> conv1.conv：

- nominal=2304。
- mean=931.89453125。
- CV=0.1836473747。
- min=692。
- max=1519。

conv1.activation -> conv2：

- nominal=2304。
- mean=956.5546875。
- CV=0.1356621811。

conv3.activation -> conv4.conv：

- nominal=2304。
- mean=1893.6328125。
- CV=0.0171073234。

conv4.activation -> conv5：

- nominal=2304。
- mean=1889.75。
- CV=0.0133975277。

上述真实训练结果说明名义扇出与有效扇出的差异已在20轮模型中出现。

但原论文明确说明采用初始结构常数以缓解反馈问题，不能将这种设计描述为作者没有认识到的缺陷。

H-016保持PARTIALLY_SUPPORTED，补充真实早期训练模型的描述性证据；对训练稳定性和任务性能的影响仍未知。

H-017保持UNTESTED。

## 4. 新科研假设H-018

H-018：

在U2相对于U0的联合稀疏训练中，连接稀疏化产生负向计算成本变化，而活动重新分布可能抵消部分收益。

数学预测：

Delta_C_W < 0。

Delta_C_S > 0。

当前状态UNTESTED。

全局FR上升不等于实际活动成本项必然为正。

本轮通过相同样本上的反事实成本计数直接验证上述预测。

本轮只进行成本归因，不宣称能够建立权重惩罚对发放变化的因果关系。

## 5. R69—R73理论公式

设同一训练轮次、同一验证样本上：

M0为U0实际有效连接掩码。

M2为U2实际有效连接掩码。

s0为U0实际各监控卷积输入的二值脉冲活动。

s2为U2实际各监控卷积输入的二值脉冲活动。

采用R53—R56及实际SOPMonitor相同统计范围。

R69：

C00=C(M0,s0)。

C20=C(M2,s0)。

C02=C(M0,s2)。

C22=C(M2,s2)。

C00/C22是各自模型实际前向产生的SOPs。

C20/C02只是在固定真实输入活动下替换计数掩码，不代表真实交叉模型前向。

R70：

Delta_C_W=0.5*((C20-C00)+(C22-C02))。

R71：

Delta_C_S=0.5*((C02-C00)+(C22-C20))。

R72：

Delta_C=C22-C00=Delta_C_W+Delta_C_S。

R73：

I_WS=C22-C20-C02+C00。

该分解为两种排列平均的代数归因，类似两变量Shapley归因思想。

不是新剪枝算法。

不代表训练因果效应。

## 6. 固定训练检查点

从EXP-017原第20轮latest检查点继续。

U0：

experiments/EXP-017/runs/EXP-017-PILOT-001/U0/latest.pth

SHA256：

1594cbb3f65c1acf91efcbe9c0a842c9ad08fbb57dbfac2ee9333afa951439c2

U2：

experiments/EXP-017/runs/EXP-017-PILOT-001/U2/latest.pth

SHA256：

c4d5d9e882282c1e18ebae3134ab6a1bcbc661acb24035087d1c44f62cbc464c

U1备用：

experiments/EXP-017/runs/EXP-017-PILOT-001/U1/latest.pth

SHA256：

bbf3a5dc7c0035bbaf8c08a18f2143bbcccc920fba6e2516b86b69624f327f81

本轮不续训U1。

三个父检查点必须保持只读，禁止覆盖。

新训练检查点保存在EXP-018独立Run ID下。

## 7. 固定训练配置

复用EXP-017全部原始训练协议。

- CIFAR-10。
- CifarNet。
- base_channels=256。
- T=4。
- 官方CuPy LIF。
- tau=2。
- v_threshold=1。
- v_reset=0。
- detach_reset=True。
- ATan替代梯度。
- batch_size=256。
- num_workers=4。
- seed=12450。
- SGD。
- Momentum=0.9。
- 初始lr=0.1。
- weight_decay=0。
- Cosine调度周期300轮。
- min_lr=0。
- warmup_epochs=0。
- cooldown_epochs=0。
- AMP=True。
- accumulation_steps=1。
- TET=False。
- label_smoothing=0。
- Mixup及CutMix关闭。

锁定：

U0：

lambda_weight=0。

lambda_neuron=0。

U2：

lambda_weight=8.834108737251828e-08。

lambda_neuron=8.834108737251828e-08。

不得重新计算或改变lambda。

不得从第21轮重新开始300轮scheduler。

完整训练总轮数仍为300。

## 8. 固定数据及随机过程

继续使用：

experiments/EXP-004/splits/indices.json

SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

训练集45000张。

验证集4500张。

500张旧科研诊断池继续隔离。

只读取data_batch_1至data_batch_5。

不读取官方test_batch。

复用ROUND-016的逐轮采样逻辑：

sampler_seed=12450+epoch。

worker_seed=99912450+epoch。

每轮固定175个训练Batch。

每轮使用44800张样本，丢弃200张尾批样本。

同轮U0与U2必须使用相同训练样本顺序和可追溯随机增强控制。

每轮保存采样顺序哈希及丢弃样本索引。

保留数据增强、归一化和验证预处理与EXP-017一致。

## 9. 环境及原始代码保护

使用已验收的：

.local/round014-venv

先确认实际解释器路径。

继续使用：

- PyTorch 2.4.1+cu118。
- CuPy 13.5.1。
- SpikingJelly 0.0.0.0.14及已记录的np.int兼容补丁。
- timm 1.0.22。
- NumPy 1.26.4。

保持vendor/UniSparse官方源码不变。

复用：

experiments/ROUND-016/common.py

experiments/ROUND-016/train.py

experiments/ROUND-016/recovery_audit.py

新增必要的ROUND-017续训及成本诊断模块。

依据实际代码决定具体修改范围。

必须将本轮预算、输出路径、Run ID及续训入口与EXP-017分离，禁止复用旧实验目录作为写入目标。

## 10. 正式续训前置检查

必须完成：

1. 父检查点SHA256验证。
2. 实际源码、环境及数据哈希核对。
3. 模型参数及BN运行状态恢复。
4. 权重阈值及稀疏buffer恢复。
5. 优化器及Momentum状态恢复。
6. 学习率scheduler状态恢复。
7. AMP scaler状态恢复。
8. Python、NumPy、Torch、CUDA RNG状态恢复。
9. 下一轮采样种子为第21轮对应值。
10. U0/U2实际lambda与锁定值一致。
11. 评估状态隔离通过。
12. 原始更新顺序一致。

检查第21轮学习率与EXP-017保存的scheduler状态及300轮计划相符。

不得因恢复训练而重置Momentum或AMP scaler。

使用独立临时模型进行短程GPU恢复检查。

建议对U0和U2各使用固定的相同真实训练Batch，对照：

A：加载父检查点后直接执行成功更新。

B：加载相同父检查点、保存并恢复临时完整状态后执行相同更新。

比较模型参数、阈值、优化器动量、scheduler及scaler状态。

逐项记录差异、容差及非确定性来源。

顺序执行以避免同时驻留两套完整模型造成显存不足。

该测试状态不得用于正式续训。

正式续训重新加载原始父检查点。

通过该检查也不能宣称跨设备、跨环境的全部长期训练轨迹逐位一致。

若关键恢复状态错误，则停止EXP-018，不重新开始完整训练。

## 11. 保留作者原始训练更新机制

继续使用原始：

SparseConv2d。

SparseBN。

SparseNeuron。

calc_penalty。

penalize。

clear_penalty。

官方CuPy LIF。

保持EXP-017已通过检查的更新顺序：

任务前向及反向。

functional.reset_net。

optimizer/scaler step。

scaler.update。

optimizer.zero_grad。

对于U2：

calc_penalty。

penalize(model,lambda_weight*lr,lambda_neuron*lr)。

clear_penalty。

对于U0保持零惩罚路径。

保留原作者AMP跳步后手动惩罚仍执行的行为。

记录每轮实际成功optimizer更新数、AMP跳步数、手动更新次数及阈值状态。

禁止加入动态有效扇出惩罚、R64归一化惩罚、蒸馏、预算损失或其他候选创新机制。

## 12. EXP-019成本计数实现

先于正式长时间续训，对第20轮U0/U2检查点完成必要的公式和实际模型成本验证。

仅在eval及no_grad模式下执行。

对每个实际被SOPMonitor监控的SparseConv2d，获取：

M0_l=U0的真实二值sparse_mask。

M2_l=U2的真实二值sparse_mask。

x0_l=U0该层真实输入。

x2_l=U2该层真实输入。

要求：

- 两组架构及输入输出形状一致。
- 对应卷积参数一致。
- stride、padding、dilation、groups一致。
- 样本顺序完全配对。
- 实际卷积输入为二值脉冲。
- 保留全部实际时间步。
- 不使用神经元输出的简单全局FR替代卷积实际输入。
- 不对非二值输入未经说明套用二值事件计数。

注意：

池化和flatten已经包含在真实后继卷积输入中。

因此本轮成本分解直接使用消费者输入，不需要将池化前源脉冲近似映射为后继卷积输入。

对于非脉冲首层和作者net.skip排除的层，保持作者原统计范围。

计算每层：

C00_l=sum(conv2d(x0_l,M0_l))/实际样本数。

C20_l=sum(conv2d(x0_l,M2_l))/实际样本数。

C02_l=sum(conv2d(x2_l,M0_l))/实际样本数。

C22_l=sum(conv2d(x2_l,M2_l))/实际样本数。

将[T,B,C,H,W]正确转换为卷积输入布局，或复用作者snn_to_ann_input。

累加所有实际Batch的原始计数，最后除以真实总样本数。

不得对不同大小Batch简单平均。

全网络Cij为各层Cij之和。

建议通过监控器hook在单次模型前向中同时计算本组掩码成本与另一组掩码成本。

可以顺序加载、评价U0和U2。

仅保存必要的对应掩码。

禁止保存全部验证样本的完整脉冲活动张量。

禁止在计数时实际改变模型前向权重或活动。

所有反事实组合只是计数，不产生新的任务logits。

## 13. R69—R73正确性检查

先使用CPU小型确定性人工实例验证：

1. C00/C20/C02/C22与独立显式连接枚举一致。
2. C00=C22时不一定意味着两个归因项均为0。
3. M0=M2时Delta_C_W=0。
4. s0=s2时Delta_C_S=0。
5. R72分解恒等式成立。
6. I_WS计算正确。
7. 不完整Batch的样本加权正确。
8. 卷积边界、stride及padding处理正确。

人工整数计数要求精确一致。

真实模型的C00/C22必须与独立原作者SOPMonitor结果在相同输入、掩码、精度及统计范围下核对。

建议归一化误差：

abs(actual-reference)/(1+abs(reference)) <= 1e-5。

对误差原因进行审查，不能通过放宽容差掩盖计数错误。

计数使用整数安全累加或足够精度的浮点累加，避免大规模SOPs求和溢出。

保存绝对误差和相对误差。

若C00/C22无法可靠对应作者Monitor，不开展正式成本归因。

## 14. EXP-019正式测量及判定

预先固定评价轮次：

- 第20轮。
- 第60轮。
- 第100轮。
- 第200轮。
- 第300轮。

只有对应轮次的U0和U2检查点均有效时才执行。

使用原4500张验证样本。

本轮使用已参与模型选择的验证集，不能称为独立泛化评价。

每个实际轮次记录：

- 逐层C00。
- 逐层C20。
- 逐层C02。
- 逐层C22。
- 逐层Delta_C_W。
- 逐层Delta_C_S。
- 逐层I_WS。
- 全网络对应指标。
- 相对C00归一化比例。
- 各层有效权重密度。
- 实际输入脉冲率。

定义：

R_W=-Delta_C_W/C00。

R_S=-Delta_C_S/C00。

R_total=(C00-C22)/C00。

R_total=R_W+R_S。

若R_S为负，表示活动变化抵消部分结构收益。

H-018主要方向预测：

Delta_C_W < 0且Delta_C_S > 0。

本轮预先指定第60轮作为主要早期训练机制诊断点。

第20轮作为已有模型诊断，其余轮次作为纵向观察。

建议实用性探索门槛：

Delta_C_S/C00 >= 0.05。

即活动重分布导致的额外成本至少达到参考计算量的5%。

这属于预先指定的探索性幅度，不是理论必然阈值，也不是统计显著性标准。

若Delta_C_S <= 0，则当前方向预测不获得支持。

若Delta_C_S为正但小于5%，报告方向符合预测但实用幅度不足。

若不同轮次结果不一致，保存真实变化，不选择单个有利轮次作为普遍结论。

记录全部结果。

不得由此直接将H-017标记为SUPPORTED。

## 15. EXP-018分阶段续训计划

训练分为：

阶段A：

U0、U2分别从第21轮续训至第60轮。

阶段B：

两组分别从第61轮续训至第100轮。

阶段C：

两组分别从第101轮续训至第200轮。

阶段D：

两组分别从第201轮续训至第300轮。

每个阶段均需两组达到相同轮次后比较。

每个阶段保存独立固定轮次检查点。

建议按U0后U2顺序执行每个阶段，并在阶段切换时正确释放GPU资源和恢复模型状态。

阶段A/B/C完成后进行进入下一阶段的检查。

全部阶段继续条件：

1. U2相对同轮U0的验证准确率下降不超过5个百分点。
2. U2相对同轮U0的SOP_sample_weighted降幅至少5%。
3. U2每个受惩罚层有效连接密度至少1%。
4. 两组损失、参数、阈值、必要梯度及手动更新保持有限。
5. 两组实际完成轮数一致。
6. 检查点、数据及训练协议有效。
7. 当前剩余资源预算能够支持下一阶段或其可安全保存的部分。

若全部门槛通过，在用户已授权预算内继续下一阶段，无需重复请求批准。

若任一门槛未通过：

停止进一步正式续训。

保留所有真实结果。

记录失败门槛、实际数据及可能原因。

不擅自降低lambda。

不从U1自动开始训练。

不延长训练轮数或更改评价标准。

若成本分解代码出现独立诊断问题，先区分是否影响原训练与原SOPMonitor正确性。不得将失败的诊断数据当作有效；若不能在预算内安全排除影响，则停止后续任务并报告。

## 16. 训练全过程风险监控

除阶段性门槛外，保持正常工程停止条件。

每轮记录：

- train loss及accuracy。
- validation loss及accuracy。
- 实际lr。
- 全部有效权重稀疏率。
- 各受惩罚层稀疏率。
- 各层权重阈值。
- 各层原始权重绝对值分位数。
- BN gamma/beta均值及离散程度。
- 全局及逐层FR。
- SOP_author。
- SOP_sample_weighted。
- optimizer成功与AMP跳步。
- 稀疏更新次数。
- CPU及GPU资源。
- 检查点状态。

建议增加只读统计：

每层threshold / median(abs(raw_weight))。

如果分母为0，记录NA及实际原因。

同时保留R66理论累计阈值与实际阈值变化比较。

R66为精确算术、固定形状惩罚及实际更新次数下的预测。

不能把初始化权重分布直接当作后期实际权重分布。

如某个关键受惩罚层有效连接密度为0并持续两个完整Epoch，停止对应运行。

出现非有限参数、非有限手动更新、不可恢复异常或OOM，立即停止受影响运行。

正常AMP跳步不自动等同于训练失效。

## 17. 有效扇出及后续理论研究准备

第20、60、100、200、300轮分别记录：

- 各层名义扇出。
- 通道有效扇出均值。
- 标准差。
- CV。
- min/max。
- P10/P50/P90。
- 零有效扇出通道比例。
- 对应完整有效扇出向量。
- 逐层活动率。
- 对应计算成本变化。

优先延续EXP-017四个直接消费路径。

对有MaxPooling或flatten的路径，分别记录真实消费者输入活动；不要将R59的直接通道近似强行应用到源神经元。

R64归一化有效扇出仍只作为理论候选，不参与正式优化。

可根据上述真实数据判断通道间有效代价差异是否随训练加剧。

不根据观察结果调整正在运行的原算法。

## 18. 检查点及续训状态管理

每组保留：

- latest.pth。
- best_validation.pth。
- epoch-060.pth。
- epoch-100.pth。
- epoch-200.pth。
- epoch-300.pth。

仅为已实际完成的轮次保存对应固定检查点。

原EXP-017第20轮检查点保持不变。

每个完整Epoch保存可恢复latest状态。

原子写入模型、optimizer、scheduler、scaler、RNG、训练轮次、数据采样状态及真实配置。

继承EXP-017历史最佳验证指标与对应检查点记录。

最佳检查点只能作为次要评价结果。

第300轮固定检查点作为本轮主要完整训练结果，不得以某个更有利的中途轮次替代。

对每个固定轮次检查点保存大小、路径、SHA256、父Run ID、实际代码版本及完成轮数。

如果发生中断，保留已完成的完整Epoch。

不得将未完成Epoch标记为已完成。

如需从最新完整Epoch重新执行中断轮次，使用唯一新运行段ID并保留失败记录。

检查点权重留在服务器，不上传GitHub。

## 19. 原始数据及大型CSV管理

完整训练原始逐样本数据可能超过GitHub单文件限制。

必须采用可追溯的分组或分片管理。

建议：

- epoch_metrics.csv保存全程汇总。
- validation_per_sample/U0/按Epoch分片保存。
- validation_per_sample/U2/按Epoch分片保存。
- validation_batch_sops按组及轮次保存。
- cost_decomposition按检查点轮次保存。

避免生成不可提交的超大单一CSV。

可以使用压缩数据或经过批准的外部存储。

必须记录路径、样本索引、数据行数、格式、SHA256和读取方式。

所有正式验证样本的结果均应保留，不得仅保存最佳Epoch或有利结果。

若存储资源不足，停止相关新增大文件任务并报告，不能静默丢弃。

## 20. 资源预算

本轮需要用户新授权。

建议：

- 1张符合实验室共享管理规定的RTX4090。
- GPU总占用上限8小时。
- CPU累计计算上限16小时。
- 正式训练最多U0/U2各280个新增Epoch，共560个新增Epoch。
- 优先选择实际可用显存不少于18 GiB的GPU。

EXP-017实际峰值reserved约15.43 GiB，应保留显存余量。

不得使用未经授权的第二张GPU。

启动前重新检查共享GPU实际负载、可用显存、磁盘空间及CPU资源。

不得终止、暂停或干扰其他用户的进程。

复用可靠持久化进程管理、心跳监控、资源截止和原子检查点机制。

历史工程估算：

U0平均每轮训练、验证及保存约34.42秒。

U2平均每轮约46.34秒。

剩余560轮合计约6.28小时。

该估计不包含全部额外诊断、启动、哈希审计及共享负载波动。

GPU预算需包含前置检查、恢复测试、训练及机制诊断。

记录资源开始时间、截止时间和实际GPU UUID。

不得因进程中断或重新启动而重置累计资源预算。

如CPU子进程统计存在覆盖不全，明确记录局限，并采用保守预算控制。

预算临近时优先保存完整可恢复检查点并安全停止。

## 21. 科研假设状态

H-016：

PARTIALLY_SUPPORTED。

固定掩码合成机制得到支持。

EXP-017第20轮真实训练模型已经出现名义与实际有效扇出差异。

对实际任务性能、梯度分配和稳定性的影响尚未确认。

H-017：

UNTESTED。

本轮不实现有效连接感知活动惩罚。

H-018：

UNTESTED。

按R69—R73正式测量四项成本。

Codex报告预设条件是否满足，但不自行作出超出实验范围的最终科学结论。

所有历史AT-LIF假设保持原状态及适用边界。

## 22. 完整训练后主要评价

若EXP-018完成至第300轮，报告：

U0/U2固定第300轮：

- 验证准确率。
- 验证CE。
- 全局及各层FR。
- 作者SOPs。
- 样本加权SOPs。
- 权重稀疏率。
- 真实非零有效权重数量。
- 总参数量。
- 各层最低有效连接密度。
- 训练时间及显存。
- 完整300轮AMP及手动更新统计。

次要报告：

历史最佳验证准确率检查点。

如果第300轮U2相对U0损失过大，仍然保留真实固定终点结果，不事后使用其他轮次宣称完整训练成功达到性能目标。

完成受控训练不等于精确复现论文表格。

论文正式非零lambda仍未确认时，使用明确名称：

Controlled UniSparse Baseline。

本轮没有独立训练种子，不宣称统计稳定性。

不访问官方测试集。

## 23. 科研状态更新

新增：

research/ROUND-017-TASK.md

更新：

- THEORY.md，追加v0.2.3及R69—R73。
- HYPOTHESES.md，补充H-016真实模型证据，新增H-018。
- RESEARCH_STATE.md。
- UNISPARSE_BASELINE.md。
- EXPERIMENT_PLAN.md。
- EXPERIMENT_REGISTRY.csv。
- DECISION_LOG.md，追加DEC-017。
- LITERATURE.md，记录对称两变量成本归因不作为原创算法。

EXP-017保持COMPLETED。

EXP-018及EXP-019分别按真实执行情况记录。

不能因EXP-018完成训练就自动把EXP-019标记为完成。

若阶段性门槛未通过，区分成功执行阶段性实验与完整300轮训练未完成。

完整训练未达到300轮时，EXP-018按实际状态记录PARTIAL或其他适用枚举，并说明原因。

## 24. 结果交付

建立experiments/EXP-018和experiments/EXP-019独立目录。

至少包括：

- TASK.md
- config.json
- environment.txt
- code_version.json
- commands.sh
- status.json
- parent_checkpoint_manifest.json
- resume_preflight_tests.json
- recovery_audit.json
- epoch_metrics.csv
- layer_sparsity.csv
- threshold_trajectory.csv
- raw_weight_statistics.csv
- bn_statistics.csv
- effective_fanout_statistics.csv
- training_sample_audit.csv
- validation_batch_sops
- validation_per_sample
- stage_gate_results.json
- resource_usage.csv
- checkpoint_manifest.json
- cost_decomposition_tests.json
- layer_cost_decomposition.csv
- total_cost_decomposition.csv
- cost_count_audit.json
- summary.csv
- REPORT.md
- limitations.md
- run.log
- figures/

按两个实验实际职责合理放置文件，不要求无关内容重复保存。

保存每次真实执行的Run ID和代码SHA。

图表至少包括：

1. U0/U2完整可用训练曲线。
2. 两组精度—SOPs权衡。
3. 各层连接密度随训练变化。
4. 阈值与原始权重分布的关系。
5. 各层有效扇出变化。
6. 四种反事实SOPs成本。
7. 连接项与活动项的成本分解。
8. 不同轮次活动成本抵消或协同效应。

未完成项目标记NA及真实原因。

## 25. GitHub交付及最终返回

正式运行前提交任务、固定配置、恢复入口及成本分解实现。

每个阶段完成后保存真实状态，使用不影响运行代码版本可追溯性的方式提交结果。

若训练进程依赖Git HEAD一致性，应采用独立日志工作区或其他安全方案，避免提交结果时改变实际运行代码身份。

最终推送全部真实结果。

不修改main。

不force push。

不删除历史失败记录。

最终返回：

- GitHub实验分支。
- 实际执行代码SHA。
- 结果提交SHA。
- EXP-018执行状态。
- EXP-019执行状态。
- U0/U2各实际完成轮数。
- GPU恢复前置检查结果。
- 各阶段门槛结果。
- 各阶段真实准确率、SOPs、FR和稀疏率。
- C00/C20/C02/C22。
- Delta_C_W、Delta_C_S及I_WS。
- 逐层成本分解结果。
- H-018方向及探索性幅度门槛结果。
- 最终检查点路径及SHA256。
- 实际CPU和GPU资源消耗。
- 未完成项目。
- 全部异常、限制及可能的理论问题。
- 实验报告与原始数据路径。

本轮任务是完成原UniSparse的可靠Controlled Baseline，并取得真实连接—活动成本归因证据。

禁止在原Baseline训练尚未完成科学审查前启动新的创新算法。

获得用户资源授权后执行。

在用户批准的总预算内，若阶段性门槛全部通过，自动继续下一阶段，不需要重复申请授权。

