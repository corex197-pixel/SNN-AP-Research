# Codex 科研执行任务：ROUND-009

# AT-LIF 高活动稀疏率下的计算预算约束微调与自蒸馏

## 一、任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-009
- 当前阶段：阶段3，候选改进方法最小实现及性能先导验证
- 科研决策：DEC-009，PIVOT + IMPLEMENT
- 理论版本：v0.1.5
- 新科学问题：Q-006，高稀疏AT-LIF的预算约束性能恢复
- 新科研假设：H-007、H-008
- 实验编号：EXP-008
- 候选方法工作名称：SCSD-AP
- GitHub仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-008-cost-aware-pruning
- 本轮建议分支：research/round-009-budget-distillation
- 上轮执行代码SHA：846c4b49745dbb2e7fae717d03eb5836609313e8
- 上轮结果SHA：199451fe48f931a5e79cad8e97fb5db62804e414
- 依赖实验：EXP-004、EXP-007
- 优先级：P0 SOPs代理正确性；P1固定协议微调；P2统计与科研交付

首先读取AGENTS.md、THEORY.md、HYPOTHESES.md、RESEARCH_STATE.md、ROUND-008-TASK.md、EXP-004及EXP-007的真实实验记录。

本轮允许短周期微调，但禁止重新训练200轮、超参数搜索、额外数据集实验或未经批准增加实验组。

## 二、ROUND-008正式验收结论

EXP-007-RUN-001因B2循环实现错误标记INVALID，保留历史。

EXP-007-RUN-002有效完成。

真实验证结果：

B0：

- Accuracy=0.9171111111111111
- Loss=0.28307441925777166
- SOPs=55023981.39022222

B1，全层阈值×1.15：

- Accuracy=0.9117777777777778
- Loss=0.28050099616934693
- SOPs=47501319.424
- SOPs降幅=0.13671606045503282

B2，静态排序：

- Accuracy=0.9008888888888889
- SOPs=48753861.91644444

M1，动态边际评价：

- Accuracy=0.8993333333333333
- SOPs=49262350.62755556

M1相对B2的准确率差为-0.1556个百分点，95%配对CI=[-1.00,+0.67]个百分点。

M1未达到预注册性能筛选门槛。

B1在实际观测的评价点上同时具有比B2、M1更高的准确率和更低的SOPs。

H-006正式判定INSUFFICIENT_EVIDENCE。

停止推进当前动态贪心搜索实现，但不将一次结果扩大为该类方法普遍无效。

保留原有H-001至H-005全部科学状态及历史实验。

## 三、本轮研究目标

ROUND-008已经证明简单统一阈值提高具备实际活动压缩价值。

但此前全部方法均冻结权重，网络无法通过梯度训练适应新的活动状态。

本轮研究：

1. 短周期微调能否恢复进一步活动剪枝造成的性能损失？
2. 微调是否引起SOPs回升？
3. 显式SOPs预算约束能否缓解活动回升？
4. 自蒸馏是否能在相近计算预算下带来额外任务性能收益？

本轮借鉴成熟知识蒸馏与计算量正则化思想，不预先宣称原创性。

本轮只验证最小方法的实际价值。

## 四、科研假设

H-007，UNTESTED：

固定提高后的阈值，普通任务微调可能引起全网络SOPs回升；引入显式SOPs预算约束能够缓解该问题。

H-008，UNTESTED：

在相近实际SOPs预算下，自蒸馏能够使高活动稀疏率AT-LIF获得比非蒸馏微调更好的任务性能。

两项假设均限定当前固定训练模型、数据和先导微调协议，不推广至其他训练种子和架构。

## 五、模型与固定数据

使用EXP-004第194轮最佳检查点：

experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth

SHA256：

c068f54379cb5d29bcfa5fe2d9de676cd3cc153433ada1ed5aa1c66355d0b32a

数据划分：

experiments/EXP-004/splits/indices.json

SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

使用原45000张训练样本和4500张验证样本。

沿用原训练数据增强、顺序采样器和验证预处理。

原500张科研诊断池不参与本轮训练或评价。

禁止使用官方测试集。

使用ROUND-008已固定的500张校准集：

experiments/EXP-007/calibration_indices.json

SHA256：

b849b74ef19cb4a09df756c54871122f3b02aa047c704d0436feab15a38bde52

该校准集仅用于SOPs归一化基准及必要计数验证，不重新开展阈值搜索。

原模型校准SOPs：

C0=54945117.504/样本。

校准集上全层阈值×1.25曾得到20.632527%的SOPs降幅。

因此本轮固定学生初始阈值倍率1.25，不根据新验证结果重新选择。

如果实际校准基准重新测量存在差异，记录真实差异及数值精度条件，不静默修改已锁定的C0。

若出现超出预设数值容差的实质性差异，停止并检查实现。

## 六、模型初始化及训练状态

教师模型：

- 使用原始第194轮检查点。
- 原权重、阈值、BN状态不变。
- eval模式。
- no_grad。
- 禁止阈值更新。
- 保证教师的前向活动统计不会污染后续状态。
- 教师与学生不得共享可被修改的参数或buffers。

学生模型：

- 从相同原检查点独立复制。
- 所有19个ATLIF阈值乘以1.25。
- 全部阈值requires_grad=False。
- 禁止手动threshold_update。
- 允许非阈值参数微调。
- 允许BN affine及运行统计按正常训练方式更新。
- 保持原始硬重置、tau、替代梯度及前向动力学。

建议在所有微调组中设置ATLIF活动手动更新系数sp=0，并确认这一设置不改变原前向输出及非阈值任务梯度。

这属于本轮明确的新训练协议，不得描述为原作者完全相同的阈值训练策略。

不能只冻结requires_grad而仍执行手动阈值更新。

每组从同一原始学生初始化独立开始。

不得将前一实验组微调完成的模型作为后一实验组的起点。

## 七、任务损失

原模型输出逻辑维度[T,B,C]。

使用TET=False的原任务定义：

z[b,c]=mean_t(logits[t,b,c])

L_CE=CrossEntropy(z,label)

禁止使用逐时间步CE平均替代时间平均logits的CE。

## 八、知识蒸馏公式

R24：

temperature=3

p_teacher=softmax(z_teacher/temperature)

log_p_student=log_softmax(z_student/temperature)

L_KD=temperature^2 * KL(p_teacher || p_student)

PyTorch建议使用：

F.kl_div(
    F.log_softmax(z_student.float()/temperature,dim=-1),
    F.softmax(z_teacher.float()/temperature,dim=-1),
    reduction="batchmean"
) * temperature**2

教师logits必须detach或在no_grad中计算。

教师与学生接收完全相同的本Batch增强后图片。

禁止教师使用不同随机增强的图像。

beta=0.5。

计算KD和CE时采用数值稳定实现。

本轮不加入特征蒸馏、膜电位蒸馏或时间步蒸馏。

## 九、可微SOPs统计

### 9.1 原统计定义

实际代码：

Activity-Pruning-SNN-main/utils/utils.py

SOPMonitor对指定卷积输入执行：

x_binary=(x>0)

通过全1卷积核统计活动输入对应的连接运算数量。

排除model.skip指定的首层输入卷积。

不包含未监控的线性层、非卷积路径和实际零权重压缩收益。

原Monitor在no_grad中执行，不可直接作为可微训练损失。

### 9.2 R25训练代理

AT-LIF原始输出：

o[l,t,b,i]=theta[l]*s[l,t,b,i]

固定正阈值时：

s_proxy=o/theta.detach()

张量逻辑布局[T,B,C,H,W]。

对每个被监控卷积k，确定其实际输入来自哪个AT-LIF模块。

定义fanout[k,c,h,w]为该输入位置在此卷积中连接到所有有效输出位置的次数。

根据真实卷积kernel、stride、padding、dilation、groups和输入输出形状计算。

可以利用全1卷积核对全1输出的转置卷积或等价方法得到fanout。

注意处理output_padding及原输入尺寸。

必须验证计算结果实际覆盖正确的输入空间。

将同一AT-LIF输出所供给的多个被监控卷积分支的fanout相加。

不可漏掉shortcut卷积，也不可重复计数。

每个模块的fanout保持常量，不参与学习。

对当前Batch：

C_proxy=sum_l(sum_t,b,i(fanout[l,i]*s_proxy[l,t,b,i]))/B

其中B为实际Batch样本数。

注意：

- s_proxy不能detach。
- theta和fanout必须detach或为常量。
- 不使用(x>0).float()直接构造训练损失，否则梯度中断。
- 不得将整个C_proxy转换为item后再加入loss。
- 统计累积至少使用FP32。
- 不能额外执行改变BN或随机状态的学生训练前向。
- 按实际Batch Size归一，包括最后一个不完整Batch。
- 优先使用轻量hook或实际前向已有张量。
- 只保留当前Batch必要的计算图，及时清理引用。

若实际卷积输入并非对应AT-LIF的正阈值缩放输出，则不得强行应用上述公式。

需要先报告实际结构差异和理论适用性。

### 9.3 代理正确性测试

必须在正式训练前完成：

1. 人工构造小型卷积，分别验证padding、stride、groups及边界fanout。
2. 对当前真实ResNet20，核查所有被监控卷积和对应AT-LIF来源。
3. 比较原SOPMonitor与C_proxy在相同前向下的数值。
4. 在alpha=1和alpha=1.25分别测试。
5. 覆盖完整Batch与不完整Batch。
6. 验证残差分支计数。
7. 验证代理加入前向统计不会改变分类输出。
8. 验证超过预算时，SOP损失对至少部分相关非阈值参数产生有限、非零梯度。
9. 验证未超过预算时平方hinge梯度为0。
10. 验证教师及阈值不发生参数更新。

计数相对误差预设容差1e-5。

记录绝对误差及相对误差，不只报告通过布尔值。

有限差分不能直接验证真实硬阈值网络替代梯度的正确性，应区分前向计数等价与代理反向梯度测试。

如果SOP代理不能正确反映原计数口径，停止相关训练，不通过更换评价定义绕过问题。

## 十、计算预算损失

R26：

target_ratio=0.8

L_SOP=relu(C_proxy/C0-target_ratio)^2

mu=10。

目标SOPs为原校准基准的80%。

这是Batch级软约束，不是整个数据集的严格硬预算保证。

最终真实SOPs必须使用原SOPMonitor独立评价。

记录每轮代理SOPs与实际验证SOPs、超预算Batch比例及SOP损失。

不能将SOP代理值称为硬件实测能耗。

## 十一、实验分组

B0：

原始AT-LIF，不微调。

P0：

所有阈值×1.25，完全不微调。

F0：

相同P0初始化，5 Epoch普通微调。

loss=L_CE

F1：

相同P0初始化，5 Epoch预算约束微调。

loss=L_CE+mu*L_SOP

F2：

相同P0初始化，5 Epoch自蒸馏微调。

loss=L_CE+beta*L_KD

M2：

相同P0初始化，5 Epoch联合微调。

R27：

loss=L_CE+beta*L_KD+mu*L_SOP

四个微调组分别独立执行。

B0和P0不消耗训练Epoch预算。

不新增其他实验组。

## 十二、固定微调配置

- dataset=CIFAR10
- network=ResNet20
- activation=原始ATLIF
- T=8
- tau=0.9
- batch_size=128
- epochs=5，每个微调组
- seed=2030
- num_workers=4
- optimizer=SGD
- momentum=0.9
- nesterov=True
- initial_lr=0.005
- min_lr=0.0001
- scheduler=CosineAnnealingLR
- weight_decay=5e-4，沿用原非阈值参数组处理
- accumulation_steps=1
- AMP=True
- TET=False
- label_smoothing=0

使用新的优化器和调度器。

本轮是独立微调，不恢复原200轮优化器动量或原余弦调度状态。

scheduler按Epoch更新，四组使用相同规则并保存实际LR轨迹。

所有微调组重新设置相同初始训练随机状态。

保持训练样本、数据顺序、增强规则、初始化、Batch数及优化器设置相同。

如教师前向可能消耗额外随机状态，进行隔离或确认无随机副作用。

核查前几个Batch的样本ID、增强结果校验信息以及初始化哈希，以确认不同组的公平性。

如AMP发生跳步，记录实际成功优化器更新次数及差异。

不得因为某组表现不理想而修改超参数。

## 十三、执行流程

伪代码：

Load frozen baseline.

Evaluate B0 and P0 using identical protocol.

For group in [F0,F1,F2,M2]:

    Reset training seed and data pipeline.

    Clone original baseline.

    Set student thresholds to 1.25 times original thresholds.

    Freeze all thresholds and disable manual threshold updates.

    Initialize a new optimizer, scheduler and scaler.

    For epoch in range(5):

        For images,labels in training_loader:

            If KD is enabled:

                teacher_logits = frozen_teacher(images)

            student_logits = student(images)

            Compute CE.

            If KD is enabled:

                Compute KL distillation loss.

            If SOP constraint is enabled:

                Compute differentiable SOP proxy.

            Form group-specific total loss.

            Backpropagate and update non-threshold parameters.

            Record real components and activity statistics.

        Save epoch metrics and recoverable checkpoint.

        Advance scheduler.

    Lock the final fifth-epoch checkpoint.

    Evaluate on the fixed validation set.

Use actual original SOPMonitor for final inference evaluation.

Save all results.

本轮所有方法主结果固定使用第5轮检查点。

不允许根据各组验证准确率分别选择最佳微调Epoch。

中间Epoch验证允许用于训练曲线记录，但不得用于选模、调整超参数或提前选择有利结果。

## 十四、评价指标

每组至少记录：

- Validation Accuracy
- Validation CrossEntropy
- SOPs per sample
- Relative SOPs savings versus B0
- FR_global
- FR_author
- Layer-wise FR
- Layer-wise thresholds
- Parameter count
- Actual successful optimizer steps
- AMP skipped steps
- Wall-clock time
- Peak allocated/reserved memory

微调组每轮额外记录：

- Train Accuracy
- Train CE
- Train KD loss（适用时）
- Train SOP loss（适用时）
- Train C_proxy
- Train total loss
- Budget violation rate
- Learning rate

保存最终逐样本验证损失、预测及正确性。

如原Monitor仅能获取Batch总SOPs，保存真实Batch计数。

不得编造逐样本SOPs及其置信区间。

对最终准确率和损失差异采用2000次样本级配对Bootstrap，seed=2027。

同一验证样本必须严格配对。

本轮仍为单训练模型的先导实验，不能把样本级CI解释为跨训练随机种子的稳定性。

## 十五、科研假设判定

H-007：

定义微调SOPs变化：

rebound_F0=(C_F0-C_P0)/C_B0

rebound_F1=(C_F1-C_P0)/C_B0

若F0较P0回升至少2个百分点，且F1较F0至少减少2个百分点的基准SOPs，同时F1准确率相比F0下降不超过1个百分点，则报告当前协议下的初步实用性支持信号。

若普通微调本身没有明显活动回升，则不能宣称存在必须依赖SOP约束解决的问题。

若SOP约束降低计算量但造成严重准确率损失，则报告性能与计算量冲突。

上述为预设先导效应幅度，不是统计显著性证明。

H-008：

主要比较M2和F1。

近似预算匹配条件：

abs(savings_M2-savings_F1)<=0.02

在满足预算匹配时，若M2准确率至少比F1提高0.005，则报告KD带来的实用性筛选信号。

同时报告实际配对CI和交叉熵变化。

若预算不匹配，不做公平同预算优势宣称。

进一步对F0/F2和F1/M2进行比较，用于区分KD的单独作用及其在预算约束下的作用。

不得根据实验结果修改这些判定标准。

## 十六、候选方法整体筛选门槛

主要条件：

1. M2验证集SOPs相对B0下降至少20%。
2. M2验证准确率相对B0下降不超过1个百分点。
3. M2与F1实际SOPs降幅差不超过2个百分点。
4. M2验证准确率相比F1提升至少0.5个百分点。
5. 不增加部署模型参数或额外推理模块。
6. 工程及指标计算完整性检查通过。

所有条件满足时，只报告值得继续研究的先导信号。

若M2没有优势但F1表现良好，则保留更简单的F1作为后续候选。

若F0已获得相近效果且维持预算，则不能宣称SOP约束或KD具有额外必要性。

若所有方法表现不理想，则保留结果并结束本轮，不自动增加Epoch、超参数或新模块。

## 十七、资源预算及进程管理

本轮需要用户新的GPU训练授权。

建议：

- 一张符合实验室共享管理规定的RTX4090。
- GPU总占用不超过75分钟，包含必要测试、4组微调及最终评价。
- CPU计算累计预算90分钟。
- 全部微调训练共20 Epoch。
- 不开展超参数搜索。
- 不重新训练Baseline。
- 不使用官方测试集。

使用当前实际可用且获授权的单卡。

GPU隔离、进程管理、心跳、原子检查点及异常记录复用已验收机制。

如预测预算不足，优先保证各组获得相同训练轮数。

不得仅将完成度不同的组作为公平对照。

若某组未完成，保留其真实状态并标记PARTIAL。

出现非有限损失、CUDA OOM、阈值被错误修改、预算超限或数据版本不一致时停止。

保留历史记录。

## 十八、文献及创新性风险

在LITERATURE.md补充：

1. Thomas Louis, Benoit Miramond, Alain Pegatoquet, Adrien Girard. On Reducing Activity with Distillation and Regularization for Energy Efficient Spiking Neural Networks. arXiv:2406.18350, 2024. https://arxiv.org/abs/2406.18350

2. Optimizing the Energy Consumption of Spiking Neural Networks for Neuromorphic Applications. 2020. https://doi.org/10.3389/fnins.2020.00662

3. Xinyu Shi, Tong Bu, Zhaofei Yu. UniSparse: Combining Weight Pruning and Spike Sparsification in Spiking Neural Networks. ICML 2026. https://proceedings.mlr.press/v306/shi26a.html

记录：

知识蒸馏、活动正则化、SynOps优化已有研究。

本轮方法属于借鉴成熟思想的AT-LIF先导改进。

尚未证明独立学术创新性。

不得使用首次提出SNN蒸馏或首次优化SOPs等未经证实的表述。

## 十九、科研状态更新

新增research/ROUND-009-TASK.md。

THEORY.md追加v0.1.5及R23—R27。

明确区分：

- 原论文AT-LIF机制。
- 成熟知识蒸馏公式。
- 本项目对实际SOPMonitor的等价计数推导。
- 代理梯度适用边界。
- 未经验证的性能恢复假设。

HYPOTHESES.md：

保留H-001至H-005原判定。

H-006更新为INSUFFICIENT_EVIDENCE。

新增H-007、H-008为UNTESTED，等待理论研究员审查实际结果。

DECISION_LOG.md追加DEC-009。

记录停止当前动态搜索，转向预算约束性能恢复。

保留所有失败历史。

## 二十、GitHub交付

建立EXP-008独立实验目录和各组唯一Run ID。

保存：

- TASK.md
- config.yaml
- code_version.json
- environment.txt
- commands.sh
- status.json
- sop_mapping.json
- fanout正确性测试
- smoke测试原始结果
- epoch_metrics.csv
- layer_metrics.csv
- validation_per_sample.csv
- validation_batch_sops.csv
- summary.csv
- metrics.json
- run.log
- REPORT.md
- limitations.md
- figures/

图表至少包括：

1. Accuracy-SOPs实际评价点。
2. 各组训练损失曲线。
3. 微调前后SOPs变化。
4. 各组验证准确率。
5. 各层FR变化。
6. SOP代理与原计数器一致性检查。
7. KD及预算约束消融结果。

检查点保留于服务器，记录实际路径、大小和SHA256。

不直接提交大型权重。

正式执行前提交代码和预注册协议。

训练期间不得改变执行代码HEAD。

完成后同步全部真实结果至独立实验分支。

不修改main，不force push，不隐藏失败数据。

## 二十一、最终交付信息

- GitHub实验分支。
- 实际执行代码SHA。
- 结果提交SHA。
- SOP代理与原Monitor的数值误差。
- 梯度正确性测试结果。
- B0/P0/F0/F1/F2/M2全部真实结果。
- 每组实际SOPs降幅。
- 每组实际完成Epoch。
- M2与F1的配对准确率及损失差异。
- H-007、H-008是否达到预设门槛。
- 各组最终检查点路径及SHA256。
- 实际资源消耗。
- 全部异常及限制。
- 原始指标、报告和图表路径。

本轮目标是快速验证短周期预算约束微调及自蒸馏是否能够产生真实的精度—计算量权衡收益。

获得用户资源授权后，前置正确性检查通过即可直接开始正式微调。

不要再次返回只有工程准备的报告。

禁止编造数据，禁止强制实验得到正面结论。