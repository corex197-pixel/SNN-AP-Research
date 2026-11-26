# Codex 科研执行任务：ROUND-008

# 任务与计算成本感知的自适应活动剪枝先导实验

## 一、任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-008
- 科研决策：DEC-008，PIVOT + IMPLEMENT
- 理论版本：v0.1.4
- 新候选科学问题：Q-005
- 新科研假设：H-006
- 实验编号：EXP-007
- 候选方法工作名称：Task- and Cost-Aware Activity Pruning，TCAP
- GitHub仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-007-independent-mechanism
- 本轮建议分支：research/round-008-cost-aware-pruning
- 上轮执行代码SHA：04827fe2f690bf6c3b104266af0507661ff6d951
- 上轮结果SHA：97a2204d7b1738037b0ffe9404c4e78279d85578
- 依赖实验：EXP-004、EXP-002、EXP-005、EXP-006
- 优先级：P0最小方法实现及指标检查；P1性能先导实验；P2补充预算实验

首先读取AGENTS.md、THEORY.md、HYPOTHESES.md、RESEARCH_STATE.md、ROUND-006-TASK.md、ROUND-007-TASK.md，以及EXP-004、EXP-005、EXP-006的实验报告。

本轮允许实现实际候选活动剪枝方法，但不允许重新训练模型。

## 二、上轮正式科研判定

H-001：

CURRENT_EVIDENCE_AGAINST，限定原中部模块、原模型、alpha=1.5、R5及原预注册幅度标准。

H-002：UNTESTED。

H-003：INSUFFICIENT_EVIDENCE。

H-004：CURRENT_EVIDENCE_SUPPORTS，限定当前训练模型前部conv1.2及alpha=1.5条件。

独立244样本实验结果：

- R_FR=0.44413539994334267。
- mean(E)=0.16851421123169769。
- 平均任务损失增加0.14448353525521643 nats。
- 配对95% CI=[0.024203559310902463,0.27779497028195327]。
- R_E_attainable=0.6678826092172114。

H-005：INSUFFICIENT_EVIDENCE。

EXP-006结果：

- R_E=0.6678824600646457。
- R_loss=0.17241431949234431。
- G2相对于G1、G3、G4的配对任务损失收益CI均包含0。
- G2准确率低于G1。
- 任务损失恢复及逐位置补偿特异性没有得到充分支持。

正式暂停原累计输出幅值补偿主线。

保留全部理论推导和实验结果，不删除历史科研记录，不将H-005错误标记为明确否定。

EXP-002、EXP-004、EXP-005、EXP-006均为COMPLETED。

EXP-003保留原BLOCKED状态。

## 三、新研究问题及理论动机

原AT-LIF通过可学习阈值及活动稀疏化机制降低脉冲活动。

不同层具有不同活动冗余、任务敏感性、有效扇出及下游非线性响应。

ROUND-006探索性结果表明，冻结权重后提高不同模块阈值，对任务损失的影响明显不同。

然而，局部累计输出误差无法充分解释任务损失变化。

本轮研究问题：

是否能够使用真实前向传播获得的任务损失变化与全网络SOPs收益，进一步优化训练完成的AT-LIF阈值配置？

已有层敏感性剪枝、阈值校准及SynOps优化研究，本轮方法不能预先宣称具有原创性。

核心目标是快速验证候选改进的实际价值。

## 四、科研假设H-006

在相近全网络SOPs预算下，动态重新计算边际剪枝代价的阈值分配方法，比统一阈值缩放及静态边际代价排序获得更好的任务性能权衡。

初始状态UNTESTED。

若动态方法相对于简单对照没有稳定的性能或计算成本优势，则不应立即开展完整论文实验。

本轮属于方法先导实验，不是独立训练种子上的确认性实验。

## 五、锁定Baseline检查点

使用：

experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth

实际服务器路径以现有仓库和检查点清单为准。

SHA256：

c068f54379cb5d29bcfa5fe2d9de676cd3cc153433ada1ed5aa1c66355d0b32a

第194轮最佳验证检查点。

模型参数量1084605。

验证参考结果：

- Accuracy=0.9171111111111111。
- Loss=0.2835640575620863。
- FR_global=0.053539032719561445。
- FR_author=0.0468707097204108。
- SOPs=55023962.13333333/样本。

必须重新核查实际检查点及固定数据划分的SHA256。

禁止覆盖原始检查点。

## 六、数据协议

固定数据划分：

experiments/EXP-004/splits/indices.json

SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

从其中train的45000个官方训练集原始索引中，按照类别分层固定抽取500张校准图片，每类50张。

校准抽样随机种子2029。

优先采用明确记录的NumPy独立随机生成器；保存实际索引、生成算法、数据版本及SHA256。

按固定索引顺序执行校准。

校准集只用于阈值搜索，不参与梯度训练。

所有组使用相同校准集。

候选阈值配置锁定后，在原4500张验证集上评价。

验证集不得参与候选动作选择、搜索停止决策或阈值调节。

禁止根据验证结果重新搜索本轮配置。

原500张科研诊断池已全部用于前序机制研究，本轮不使用。

禁止使用10000张官方测试集。

明确校准样本曾参与原模型训练，验证集曾用于最佳检查点选择，因此本轮结果不能作为最终独立泛化证明。

## 七、实验配置

- CIFAR-10。
- ResNet-20。
- 原始AT-LIF。
- T=8。
- tau=0.9。
- 原第194轮模型。
- 推理FP32。
- Batch Size=32。
- 固定评估预处理。
- model.eval()。
- torch.no_grad()。
- 禁用梯度训练。
- 不创建训练优化器。
- 不修改权重、BN参数及运行统计。
- 不修改原替代梯度和硬重置机制。
- 不微调。

允许在显存不足时将推理Batch Size统一降至16或8，并记录实际变更。

所有评价采用原任务损失：

z[b,c] = mean_t(logits[t,b,c])

loss[b] = CrossEntropy(z[b,:],label[b])

按样本数计算准确率及平均损失。

评估过程复用已经验收的状态隔离机制。

## 八、SOPs计算要求

优先复用原作者SOPMonitor及EXP-004最终验证使用的SOPs定义。

参考：

experiments/EXP-004/round005/finish.py

原统计口径：

- 对指定卷积层注册SOP hooks。
- 将卷积输入中的正活动映射为1。
- 根据实际卷积结构统计相应连接运算。
- 覆盖全部时间步。
- 排除model.skip指定的首层模拟输入卷积。
- 不包含未纳入原Monitor的非卷积路径。
- 不考虑实际零权重造成的连接减少。
- 总计数除以实际样本数。

本轮计算必须使用完整网络前向之后的总SOPs，不能只使用受干预模块的脉冲数量估算全部收益。

确认残差路径、下游卷积、步幅和padding的计数与原Monitor一致。

若为提高搜索效率实现等价的快速SOP计数器，必须先与原Monitor进行对照，记录误差及适用范围。

未经确认不得改变主要SOPs统计口径。

原SOPs是指定范围的理论活动运算量，不是硬件实测能耗。

同时记录全网络FR及逐层FR。

## 九、候选方法数学公式

R21：

在固定原始权重W时：

min L_task(W, theta)

subject to C(W, theta) <= C_target

其中theta为19个ATLIF模块的阈值向量，C为完整网络的SOPs统计。

定义原始阈值theta0[l]。

候选阈值：

theta[l] = alpha[l] * theta0[l]

所有alpha初始为1。

采用固定候选阶梯：

alpha_levels = [1.0, 1.1, 1.25, 1.5, 2.0]

每次候选动作只允许一个模块从当前阶梯进入相邻的更高阶梯。

禁止降低阈值，禁止超过2倍原始阈值。

设当前阈值配置为v，候选配置为v_candidate。

R22：

delta_L = L(v_candidate) - L(v)

delta_c = (C(v) - C(v_candidate)) / C(v0)

score = delta_L / delta_c

其中v0是原始阈值配置。

只有delta_c>=0.005的候选动作才具有选择资格，即单次动作至少节省原校准基准SOPs的0.5%。

这是预先指定的搜索工程门槛，不是理论临界值。

如果delta_c为零或负数，不执行除法，该候选标记为无效或无收益。

允许delta_L为负，不得将实际损失下降强制截断为0。

每轮选择score最小的有效动作。

分数并列时：

1. 优先更大的delta_c。
2. 再按固定模块执行顺序选择。

采用真实完整网络前向得到delta_L和delta_c。

不使用替代梯度估计。

每次接受动作后重新计算剩余候选动作的真实边际代价。

该方法属于贪心局部搜索，不保证全局最优。

## 十、实验对照组

### B0：原始AT-LIF

原始权重及19个ATLIF阈值完全不变。

### B1：统一阈值缩放

全部19个模块使用相同alpha。

固定候选：

[1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.4,1.5,1.75,2.0]

在校准集上计算各配置的准确率、损失、FR及完整网络SOPs。

不根据验证集调整alpha。

### B2：静态边际代价排序

使用与M1相同的阈值阶梯。

在原始模型状态下，对19个模块分别执行一次1.0至1.1的局部候选调整。

根据R22得到初始固定排序。

随后按该固定排序依次尝试相邻阈值阶梯调整，循环访问允许继续提高阈值的模块。

实际执行每个候选动作前核查真实全网络SOPs节省是否达到0.005的门槛。

允许重新测量候选动作的实际计算量，但不得根据后续测得的损失变化重新排序。

记录每次实际接受或拒绝的动作。

### M1：动态边际代价活动剪枝

使用R22计算当前全部有效候选动作的边际代价。

选择最优动作后更新目标层阈值。

在新网络状态下重新评价其余候选动作。

与B2使用相同的阶梯、动作有效性要求、最大接受动作数量及计算预算记录规则。

B2和M1最多各接受12个动作。

M1第一轮的候选结果可以与B2共享，避免重复执行相同前向。

不得通过减少M1的评价样本量来制造搜索速度优势。

## 十一、主要算法伪代码

Input:

trained_model, calibration_data, target_budget

Store original thresholds and model state.

Evaluate initial task loss and total SOPs.

Initialize all alpha to 1.

For step in range(max_accepted_steps):

    Enumerate one-step threshold increases.

    For each candidate:

        Restore current accepted threshold vector.

        Apply only one candidate change.

        Evaluate full network on fixed calibration_data.

        Measure loss, accuracy, FR and total SOPs.

        Restore accepted threshold vector.

        Compute delta_L and delta_c.

        If delta_c >= 0.005:

            Compute score = delta_L / delta_c.

    If no valid candidate:

        Stop and record reason.

    Choose valid candidate with smallest score.

    Accept its threshold vector.

    Record all metrics and candidate evaluations.

    If target budget reached:

        Save selected threshold configuration.

        Stop.

Return selected static threshold vector.

要求所有临时候选干预在退出后恢复，不得使一个未接受候选的阈值污染后续候选。

任何完整模型参数变化必须仅限已接受的阈值配置。

## 十二、计算预算及配置锁定

主要目标：

相对于原校准集基准，全网络SOPs减少10%。

优先寻找实际节省位于10%至12%区间的配置。

如果存在多个有效候选且某候选能够进入目标区间，优先从进入该区间的候选中按对应方法的规则选择。

如果没有候选能够进入该区间，可以保留首次达到或超过10%节省的配置，并明确实际超出预算匹配范围。

若12个动作内未达到目标，报告真实最优已达到的节省比例，不增加搜索次数。

B1优先选取校准SOPs节省最接近10%且不低于10%的配置。

所有方法配置在校准阶段结束后锁定。

禁止通过验证集准确率选择多个候选配置中最有利的一个。

资源允许时，允许在同一搜索路径基础上继续评估20% SOPs节省目标。

20%目标属于补充探索，不得因其结果更好而替代10%主结果。

## 十三、评价指标

在4500张验证集上计算：

- 分类准确率。
- 平均CrossEntropy。
- FR_global。
- FR_author。
- 完整网络指定口径SOPs。
- 相对SOPs下降比例。
- 每层最终阈值及FR。
- 参数量。
- 校准搜索使用的候选前向数量。
- 搜索处理样本总数量。
- 校准时间。
- 最终评价时间。
- GPU峰值显存。

保存每个最终模型的逐样本损失、预测类别和正确性。

尽可能保存逐样本SOPs；如果原Monitor仅能得到Batch总量，则保存Batch原始计数并明确统计边界。

使用2000次样本级配对Bootstrap，seed=2027，计算主要损失和准确率差异的95%置信区间。

若无法计算逐样本SOPs CI，不得编造置信区间。

## 十四、先导成功标准

以下是项目内部筛选标准，不是发表门槛。

主要目标：

1. M1验证集SOPs相对于B0实际下降至少10%。

2. M1验证准确率相对于B0下降不超过1个百分点。

3. M1与B1/B2的实际验证集SOPs下降比例差距不超过2个百分点时，进行同预算近似比较。

4. 在相近SOPs条件下，重点观察M1是否优于B1和B2。

5. 若M1准确率较两种简单对照均提高至少0.5个百分点，则作为值得继续研究的实用性筛选信号。

同时报告置信区间、任务损失及完整原始结果。

不将0.5个百分点的点估计差异自动解释为统计显著。

若实际预算不能匹配，不做公平的同预算优势宣称。

若M1相对于B2没有实际增益，不能认定动态重新评价机制有效。

若只有B2或B1表现良好，应报告简单方法可能已足够。

若所有方法均无法取得合理准确率与SOPs权衡，则停止当前候选方法，不在本轮增加模块或超参数进行补救。

## 十五、必要工程检查

在正式搜索前确认：

- 模型SHA及数据划分正确。
- 实际19个ATLIF模块名称及执行顺序正确。
- alpha=1.0时与原模型输出一致。
- SOPMonitor重复使用时计数不会累计污染。
- 改变阈值后权重及BN状态不变。
- 候选评价后阈值恢复。
- 无副作用评估隔离有效。
- 原始检查点及代码未被覆盖。
- 不使用测试集。
- 新Run ID不会覆盖历史数据。

先完成少量样本冒烟测试。

不得再次重复整个EXP-001动力学单元测试。

## 十六、代码修改范围

首先检查实际仓库。

优先建立独立搜索与评价脚本，不修改原Baseline核心训练文件。

允许新增：

- 阈值配置管理。
- 完整网络SOPs评价函数。
- 校准集固定划分。
- 静态边际排序算法。
- 动态边际评价算法。
- 统计及绘图脚本。
- 必要单元测试。

输出最终19个静态阈值配置及模型元数据。

若输出新模型检查点，应保存在独立文件，不能修改原始第194轮检查点。

禁止同时加入知识蒸馏、替代梯度修改、膜电位补偿、权重剪枝等其他创新模块。

## 十七、资源预算

建议用户授权：

- 单张符合实验室共享管理规定的RTX 4090。
- GPU累计占用上限30分钟。
- CPU计算累计不超过30分钟。
- 不训练，不微调。
- 优先完成10%主要SOPs目标。
- 仅资源允许时完成20%探索目标。

设置真实截止时间和预算检查。

记录各方法实际搜索开销。

若预算不足，保存已完成的候选数据和实验状态，不追加计算资源。

如果最终评价尚未完成，标记PARTIAL，不能将校准集性能当作验证性能。

## 十八、科研记录

新增research/ROUND-008-TASK.md。

THEORY.md追加v0.1.4、R21、R22及贪心优化适用边界。

HYPOTHESES.md：

- 更新H-004为CURRENT_EVIDENCE_SUPPORTS，限定当前实验范围。
- 更新H-005为INSUFFICIENT_EVIDENCE。
- 保留H-001负面结果。
- 新增H-006，UNTESTED。
- 保留H-002、H-003原状态。

追加DEC-008：

暂停原幅值补偿主线，转向任务损失与实际SOPs联合驱动的候选活动剪枝方法。

将TCAP标记为工作名称及候选方法，不宣称已获得学术创新性证明。

LITERATURE.md补充和记录相关工作重叠风险：

- Adaptive Calibration: A Unified Conversion Framework of Spiking Neural Networks。
- UniSparse: Combining Weight Pruning and Spike Sparsification in Spiking Neural Networks。
- Optimizing the Energy Consumption of Spiking Neural Networks for Neuromorphic Applications。

现阶段仅完成初步风险排查，后续需要对最接近方法进行详细算法比较。

## 十九、结果交付

建立EXP-007独立目录及唯一Run ID。

保存：

- TASK.md
- config.yaml
- environment.txt
- code_version.json
- commands.sh
- status.json
- calibration_indices.json
- candidate_history.csv
- accepted_actions.csv
- selected_thresholds.json
- validation_per_sample.csv
- layer_metrics.csv
- summary.csv
- metrics.json
- run.log
- REPORT.md
- limitations.md
- figures/

至少绘制：

1. Accuracy-SOPs权衡图。
2. Loss-SOPs权衡图。
3. 各层最终阈值调整倍数。
4. 各层FR变化。
5. 动态边际分数随迭代的变化。
6. 静态与动态方法的搜索开销对比。

候选历史必须保存全部实际候选，包括未接受的候选。

## 二十、GitHub提交

从ROUND-007最终结果版本建立独立实验分支。

正式实验前提交任务、预设指标、固定校准索引和实际执行代码。

实验完成后提交真实数据、失败记录、科研报告及状态文件。

记录执行代码SHA与结果提交SHA。

不修改main，不force push，不上传大型模型权重和敏感凭证。

## 二十一、最终交付信息

- GitHub分支及执行代码SHA。
- 最终结果提交SHA。
- 校准样本数量及哈希。
- B0、B1、B2、M1全部真实评价结果。
- 各组实际SOPs节省比例。
- 各组验证准确率和损失。
- 最终阈值配置。
- 动态与静态方法的实际性能差异。
- 是否达到同预算比较条件。
- 原始候选评价历史。
- 各方法搜索计算成本。
- 实验状态、异常及局限。
- 是否达到本轮先导筛选门槛。

本轮不是要求产生正面结果，而是第一次直接测试一个具有明确优化目标的AT-LIF候选改进方法。

如果新方法有效，再进一步开展创新性系统查重、算法优化和论文实验；如果没有优势，应保留失败结果并及时淘汰。

获得用户方向确认及GPU授权后，请实际执行本轮任务，不要仅返回工程准备报告。