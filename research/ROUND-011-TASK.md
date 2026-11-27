# Codex 科研执行任务：ROUND-011

# AT-LIF 幅值与重置感知的阈值代理梯度优化

## 一、任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-011
- 科研阶段：阶段3，候选方法最小实现及性能先导验证
- 科研决策：DEC-011，PIVOT + IMPLEMENT
- 理论版本：v0.1.7
- 科学问题：Q-008
- 科研假设：H-011
- 实验编号：EXP-011
- 候选方法工作名称：ARTG（Amplitude- and Reset-Aware Threshold Gradient）
- GitHub仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-010-normalization-calibration
- 本轮建议分支：research/round-011-threshold-gradient
- 上轮执行代码SHA：b6d48fbe24db8ee26ea814cef02ec91673174e3d
- 上轮结果SHA：8202467fe7cc26e3b19bb82aa51a8f4e25e04410
- 依赖实验：EXP-000、EXP-004、EXP-008、EXP-009、EXP-010
- 优先级：P0 梯度及前向一致性验证；P1 三组短周期微调；P2 同预算正式评价

首先读取AGENTS.md、THEORY.md、HYPOTHESES.md、RESEARCH_STATE.md、ROUND-010-TASK.md及EXP-009、EXP-010报告。

从上轮结果提交创建独立实验分支。

## 二、ROUND-010正式科研判定

EXP-009及EXP-010均为COMPLETED。

20%目标：

- C0：Accuracy=0.908，SOPs=43652928.192。
- C1：Accuracy=0.9006666667，SOPs约43.998M。
- C2：Accuracy=0.8988888889，SOPs约43.944M。
- C3：Accuracy=0.8968888889，SOPs约43.078M。
- M3：Accuracy=0.8971111111，SOPs约43.096M。

M3相对C0准确率差为-1.0889个百分点，配对95% CI约[-1.9111,-0.3328]个百分点。

H-009：CURRENT_EVIDENCE_SUPPORTS，限定当前模型、固定500张校准图片、batch128及模式干预条件。

P0的BN模式切换造成SOPs变化-27.42%，但不能单独证明BN是ROUND-009训练性能变化的主要原因。

H-010：CURRENT_EVIDENCE_AGAINST，限定原主要20%近似预算下相对C0的性能改善预测。相对C3是否具有优势仍为证据不足。

保留30%预算下C2相对C0的+0.6个百分点探索性结果及跨零CI。

正式停止推进当前幅值补偿、静态和动态阈值搜索以及BN校准主线。

保留简单统一阈值缩放作为强对照。

所有历史负面证据和失败记录必须保留。

## 三、新科学问题

原AT-LIF：

m[t]=tau*u[t-1]+x[t]

s[t]=H(m[t]-theta)

o[t]=theta*s[t]

u[t]=(1-s[t])*m[t]

theta>0。

原始前向采用硬阈值及硬重置。

AT-LIF的theta同时控制二值发放、输出幅值及重置状态。

当前研究目标是检验更符合该计算图的阈值代理梯度能否改善活动剪枝训练效果。

不改变原始前向动力学，不增加推理参数，不增加新的稀疏化模块。

## 四、论文与实际代码核查

原论文：

Activity Pruning for Efficient Spiking Neural Networks，Tong Bu、Xinyu Shi、Zhaofei Yu，NeurIPS 2025。

原文第3.3节公式（13）：

d_tilde_o/d_m = -d_tilde_o/d_theta

= max(1-abs(m-theta)/theta,0)

实际代码：

Activity-Pruning-SNN-main/models/submodules/layers.py

Surrogate.backward中：

grad_input = grad_input * tmp

grad_thre = -(grad_input * tmp).mean()

由于grad_input已经乘过tmp，实际阈值梯度包含tmp平方。

定义：

r=m/theta

g=max(1-abs(r-1),0)

q=进入原Surrogate.backward的输出上游梯度。

R32：

grad_theta_code = -mean(q*g*g)

R33：

grad_theta_Eq13 = -mean(q*g)

R33保留了原代码的均值聚合方式，仅用于局部代理因子的论文公式对齐，不称为完整论文实现的绝对修复。

必须保留未经修改的G0原始实现。

不得将修正平方因子直接宣称为本项目创新。

## 五、候选方法完整数学公式

R34：

使用硬前向、软反向的STE构造二值脉冲。

令F为三角形替代梯度函数的积分。

F'(r-1)=g。

s_ST=stopgrad(H(r-1))+F(r-1)-stopgrad(F(r-1))

实际前向必须精确输出原二值脉冲，允许使用自定义Autograd Function避免浮点相消。

R35：

ds/dm = g/theta

ds/dtheta = -r*g/theta

R36：

o=theta*s

do/dm = g

do/dtheta = s-r*g

R37：

将幅值输出路径和二值重置路径分离。

a=loss对加权输出o的上游梯度。

b=loss对用于重置的二值脉冲s的独立上游梯度。

grad_m=a*g+b*g/theta

R38：

严格共享阈值代理梯度为：

sum(a*(s-r*g)-b*r*g/theta)

R39：

本轮为保持原梯度聚合尺度，使用：

grad_theta_ARTG=mean(a*(s-r*g)-b*r*g/theta)

均值覆盖当前时间步的实际共享阈值元素维度。

各时间步的共享阈值梯度贡献由PyTorch自动累加。

不得再次沿时间维度平均任务梯度。

R39是有意保留的优化尺度规则，不是严格共享参数链式法则。

## 六、具体实现规范

建议建立独立可选AT-LIF替代梯度实现，提供：

- original
- eq13_aligned
- artg

G0调用原Baseline Surrogate和原ATLIF。

G1仅修改原Surrogate.backward的阈值梯度局部因子：

保存原始上游梯度q。

grad_m=q*g

grad_theta=-mean(q*g)

保留原始前向及原始重置计算。

M4使用独立的自定义Autograd Function，输出：

1. 加权脉冲o=theta*s。
2. 二值脉冲s。
3. 原手动阈值更新统计量。

前向与原代码一致。

在神经元内部：

u=(1-s)*m

其中s来自自定义Function的独立二值输出。

对下游网络仍然返回o。

不能通过带梯度的o/theta.detach()重新构造M4的重置脉冲。

否则幅值梯度会混入重置路径。

M4 backward伪代码：

a = upstream_gradient_of_weighted_output

b = upstream_gradient_of_binary_reset_output

r = membrane / theta

g = clamp(1-abs(r-1),min=0)

s = binary_spikes

grad_m = a*g + b*g/theta

grad_theta = mean(
    a*(s-r*g) - b*r*g/theta
)

return grad_m, grad_theta, None

若某输出未被使用，其上游梯度按0处理。

手动阈值更新统计量保持原有非可微行为。

张量逻辑维度：

每个时间步[B,C,H,W]或实际等价状态形状。

theta为模块共享标量参数。

所有聚合严格遵守对应参数形状。

保持原始T=8时间展开、硬发放条件m>=theta、tau=0.9、初始膜电位和硬重置。

原活动稀疏化手动更新公式、系数和执行时机保持不变。

原优化器任务梯度更新及手动阈值更新分别记录。

三组统一沿用Controlled Baseline已登记的AMP跳步及手动更新策略，不允许某组单独变更。

如果实际AMP行为导致不公平的成功更新次数差异，记录并报告。

所有theta必须始终有限且严格为正。

不私自增加阈值裁剪、梯度裁剪、参数重参数化或自适应学习率。

若出现非正阈值，停止受影响运行并保存异常记录。

## 七、实验分组

G0：原代码阈值梯度，原始平方因子。

G1：论文公式对齐阈值梯度，保留原聚合尺度。

M4：R35—R39幅值与重置感知代理梯度。

三组均从原第194轮检查点独立初始化。

所有19个ATLIF阈值初始乘以1.25。

阈值恢复可学习。

原手动活动稀疏化系数sp=0.002。

eta2=0。

原始权重、BN affine及运行统计均从相同检查点恢复。

不使用ROUND-009的F0/F2微调权重。

不加入KD、SOPs惩罚项、BN重新校准或其他新模块。

## 八、固定检查点及数据

原始检查点：

experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth

SHA256：

c068f54379cb5d29bcfa5fe2d9de676cd3cc153433ada1ed5aa1c66355d0b32a

原固定数据划分：

experiments/EXP-004/splits/indices.json

SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

固定500张训练来源校准集：

experiments/EXP-007/calibration_indices.json

SHA256：

b849b74ef19cb4a09df756c54871122f3b02aa047c704d0436feab15a38bde52

训练使用原45000张训练数据及原训练增强。

验证使用原4500张验证图片及确定性评估预处理。

校准使用已有500张训练来源图片。

原500张科研诊断池不使用。

禁止使用官方测试集。

所有检查点只读加载，不修改历史文件。

## 九、必须在正式训练前完成的正确性检查

### 9.1 人工梯度测试

使用正阈值和不同归一化膜电位：

r=[-0.5,0,0.5,0.9,1.0,1.1,1.5,2.0,3.0]

测试：

- 原代码梯度与R32一致。
- G1梯度与R33一致。
- M4梯度与R37和R39一致。
- 单独幅值路径。
- 单独重置路径。
- 两条路径同时存在。
- 任意正负上游梯度。
- 未使用某输出的零梯度情况。
- 共享标量阈值的正确聚合维度。

float64人工测试归一化误差容差1e-10。

float32测试归一化误差容差1e-5。

误差定义：

abs(actual-expected)/(1+abs(expected))

### 9.2 多时间步测试

构造至少T=8的局部ATLIF序列。

使用原始硬重置动力学。

比较：

- 完整脉冲序列。
- 每时刻加权输出。
- 重置前膜电位。
- 重置后膜电位。
- 手动阈值更新统计量。
- 输入梯度。
- 共享阈值梯度。

在相同固定参数、输入及上游损失条件下，G0/G1/M4前向脉冲必须完全一致。

非阈值输入梯度应在数值容差内一致。

阈值梯度应当符合各自已定义公式，不要求三个版本相同。

M4可使用显式STE参考图与自定义Autograd Function交叉验证。

注意参考图的阈值归一化必须与R39一致，或先验证R38求和版本再验证显式归一化。

不能将硬阈值函数的有限差分当作所有STE梯度的真实标准。

可以补充验证在发放状态不变的局部区域内，幅值直接导数与真实前向变化一致。

### 9.3 真实ResNet20检查

使用真实检查点及相同固定Batch：

- 三组相同初始化哈希。
- 三组前向logits及二值脉冲一致。
- 原始SOPMonitor计数一致。
- 非阈值参数任务梯度在FP32下相对误差可接受。
- M4全部阈值梯度有限。
- 手动更新统计量一致。
- 三组实际优化器参数组及阈值是否参与优化正确。
- eval状态隔离正确。
- AMP训练步骤可执行。
- 不出现非有限损失或非正阈值。
- 不出现显著异常梯度。

补充记录各层：

原始任务阈值梯度。

G1任务阈值梯度。

M4任务阈值梯度。

M4的直接幅值项、发放状态项、重置项。

原手动稀疏更新量。

实际optimizer造成的阈值变化。

实际手动更新造成的阈值变化。

不能只比较原始梯度模长，因为动量和学习率等因素也会影响实际更新。

若发现非阈值梯度实质性不一致或前向动力学改变，停止正式训练并报告理论歧义。

所有正式前置测试必须在训练开始前完成。

## 十、固定训练协议

- CIFAR10
- ResNet20
- T=8
- tau=0.9
- batch_size=128
- num_workers=4
- 每组5 Epoch
- seed=2030
- SGD
- momentum=0.9
- nesterov=True
- learning_rate=0.005
- weight_decay=5e-4
- 采用原项目一致的参数组处理
- cosine scheduler
- T_max=5
- min_lr=0.0001
- AMP=True
- TET=False
- label_smoothing=0
- accumulation_steps=1

每组独立初始化新的优化器、scheduler和scaler。

不恢复原200轮训练的优化器动量。

每组相同数据顺序、随机增强及训练步数。

训练损失：

z=mean_t(logits)

loss=CrossEntropy(z,label)

不新增其他损失项。

原手动阈值更新按已验收的Controlled Baseline代码执行。

每轮记录可恢复检查点。

主结果固定使用第5轮，不依据验证准确率选择最佳Epoch。

中间验证仅用于描述训练动态。

## 十一、同SOPs预算评价

先评价三组未经额外阈值校准的第5轮最终模型。

这是直接训练效果。

再进行统一的预算校准，用于比较其最终Accuracy–SOPs权衡。

注意：各组的阈值已经独立学习，不得将其覆盖成原Baseline的阈值向量。

定义各组第5轮学习得到的阈值：

theta_learned[l]

评价时：

theta_eval[l]=alpha*theta_learned[l]

保持各组已经学到的相对层间阈值结构。

C0强对照使用原Baseline阈值向量进行相同全局倍率校准。

使用已有500张校准图片。

固定倍率：

[0.75,0.8,0.9,1.0,1.1,1.2,1.25,1.3,1.4,1.5,1.75,2.0]

主要目标：相对于原校准基准SOPs减少20%。

补充目标：减少30%。

基准C0_cal以本轮实际原模型测量为准。

优先选择校准SOP降幅位于[target,target+0.02]的候选。

选择最接近目标的候选，并列时选择较低alpha。

如果不存在符合窗口的候选，沿实际跨越目标的相邻区间最多执行3次中点细化。

不假设SOPs关于alpha严格单调。

不能根据校准准确率或验证准确率选择候选。

若仍不匹配，记录最接近的实际结果并标记预算匹配失败。

所有配置锁定后，在原4500张验证集进行正式评价。

使用原SOPMonitor。

不重新校准BN，不更新任何权重。

## 十二、科研假设H-011及筛选标准

H-011，初始UNTESTED：

在相同初始化、训练预算、活动稀疏化规则及近似全网络SOPs预算下，M4能够获得优于G0、G1以及简单阈值校准的任务性能。

主要20%预算下要求：

1. M4实际验证SOPs降幅>=0.20。

2. M4准确率相比原始B0下降<=0.01。

3. M4与G0/G1/C0的验证SOPs相对降幅差均<=0.02。

4. M4准确率相比G0/G1/C0均提高>=0.005。

5. 推理模型参数量及前向机制不变。

6. 所有实验有效性和梯度正确性检查通过。

同时报告：

- M4相对G0的配对准确率和损失差异。
- M4相对G1的配对准确率和损失差异。
- M4相对C0的配对准确率和损失差异。

使用2000次样本级配对Bootstrap，seed=2027。

点估计达到筛选门槛但CI跨零时，只报告先导信号。

不将样本级CI解释为跨训练种子稳定性。

如果G1优于G0而M4没有额外收益，不能把G1的公式对齐效果归为M4的创新贡献。

如果M4没有超过简单C0，不进入完整200轮训练。

30%预算结果仅作为补充探索，不替换主要20%结果。

若M4在两种预算下均无明显优势，本轮结束，不擅自增加训练轮数或改变梯度公式。

## 十三、训练机制诊断

记录每个ATLIF模块：

- threshold
- firing rate
- SOP贡献
- 任务阈值梯度
- 手动稀疏更新量
- 阈值实际更新量
- 阈值梯度为零的比例
- 非有限梯度数量

对于M4，额外记录：

- 直接幅值项a*s。
- 发放状态项-a*r*g。
- 重置项-b*r*g/theta。

分别报告有符号聚合和绝对量聚合，避免误差相消掩盖机制。

诊断建议固定在每轮预先指定的少数Batch上进行详细统计。

不得保存全部训练过程高维膜电位张量。

应尽可能流式统计，及时释放计算图。

检查任务梯度是否被原手动阈值更新或梯度归一化尺度淹没。

如果实际三组阈值几乎没有产生可分辨差异，应报告实验干预不足这一限制，不直接证明代理梯度无效。

不允许根据诊断结果在本轮私自修改梯度尺度。

## 十四、文献与创新性说明

在LITERATURE.md追加：

1. Learnable Surrogate Gradient for Direct Training Spiking Neural Networks，IJCAI 2023。

https://ijcai.org/proceedings/2023/335

2. SpQuant-SNN: ultra-low precision membrane potential with sparse activations unlock the potential of on-device spiking neural networks，2024。

https://doi.org/10.3389/fnins.2024.1440000

3. Kook et al.，Stabilizing Direct Training of Spiking Neural Networks: Membrane Potential Initialization and Threshold-robust Surrogate Gradient，2025。

https://arxiv.org/abs/2511.08708

4. ELVIA analysis of Activity Pruning for Efficient Spiking Neural Networks。

https://elvia-atlas.org/atlas/cases/neurips2025-0028-activity-pruning-for-efficient-spiking-neural-networks.html

注明第四项为尚待独立审查的外部分析，不当作正式同行评审论文。

已有可学习阈值、替代梯度优化、梯度路径分离研究。

本轮方法尚未证明独立学术原创性。

不宣称首次发现平方因子问题。

本轮重点是验证针对AT-LIF幅值和重置耦合的具体梯度设计是否具有实际性能价值。

## 十五、资源预算与终止条件

需要用户授权：

- 一张符合实验室共享管理规定的RTX4090。
- GPU总占用上限60分钟。
- CPU累计计算上限90分钟。
- 三组各5 Epoch，总15 Epoch。
- 不开展额外模型训练、超参数搜索或其他创新模块。

运行前检查共享GPU剩余显存和实际负载。

只使用单卡。

采用可靠进程管理、心跳、原子检查点和预算停止机制。

优先保证三组获得相同训练协议。

若预算不足，保存真实PARTIAL状态，不将不同完成度的结果作为主要公平对照。

遇到前向一致性失败、数学梯度检查失败、非有限损失、非正阈值、OOM、数据校验失败或资源超限时停止相关运行。

保留全部失败和异常记录。

## 十六、科研状态更新

新增research/ROUND-011-TASK.md。

THEORY.md追加v0.1.7及R32—R39。

明确区分：

- 论文公式13。
- 实际代码平方因子。
- 本项目独立STE推导。
- 为保持公平而保留的均值归一化。
- 尚未验证的性能假设。

HYPOTHESES.md：

保留H-001至H-008已有结论。

H-009更新为CURRENT_EVIDENCE_SUPPORTS，限定原诊断条件。

H-010更新为CURRENT_EVIDENCE_AGAINST，限定主要预算下对C0的改善预测；相对C3证据不足。

H-011新增为UNTESTED。

DECISION_LOG.md追加DEC-011。

停止当前后训练校准主线，转向阈值代理梯度优化。

## 十七、实验交付

建立EXP-011目录及各组唯一Run ID。

保存：

- TASK.md
- config.yaml
- code_version.json
- environment.txt
- commands.sh
- status.json
- gradient_unit_tests.json
- forward_equivalence.json
- real_network_gradient_check.json
- epoch_metrics.csv
- threshold_gradient_statistics.csv
- threshold_updates.csv
- layer_metrics.csv
- candidate_history.csv
- selection_lock.json
- validation_per_sample.csv
- validation_batch_sops.csv
- summary.csv
- metrics.json
- checkpoint_manifest.json
- run.log
- REPORT.md
- limitations.md
- figures/

至少绘制：

1. 三种阈值代理梯度函数。
2. 各组训练损失和准确率曲线。
3. 各层阈值变化。
4. 阈值任务梯度与手动更新量对比。
5. M4不同梯度项统计。
6. 各层FR变化。
7. 训练结束后原始评价点。
8. 20%及30%同预算Accuracy–SOPs评价结果。

全部真实运行数据提交GitHub。

大型检查点保留服务器，记录实际路径、大小及SHA256。

## 十八、最终交付信息

- GitHub实验分支链接。
- 实际运行代码SHA。
- 结果提交SHA。
- 三组人工梯度测试结果。
- 三组真实网络前向及非阈值梯度一致性结果。
- 三组完整训练Epoch数量。
- 三组未经校准的实际性能。
- 20%及30%目标下同预算性能。
- M4相对G0/G1/C0的配对差异及CI。
- 各组实际阈值变化。
- M4梯度各项的统计结果。
- H-011是否达到预设筛选条件。
- 全部检查点路径及SHA256。
- 实际GPU及CPU资源消耗。
- 异常与局限。
- 实验报告及原始数据路径。

执行前提交任务、配置及代码。

获得用户资源授权且所有必要前置检查通过后，直接开始正式训练。

不修改main，不force push，不覆盖原始Baseline实现。

本轮目标不是证明数学形式更复杂的梯度一定更好，而是判断符合AT-LIF幅值与重置计算图的代理梯度，能否真正改善活动剪枝效果。

禁止编造结果，禁止强制支持科研假设。