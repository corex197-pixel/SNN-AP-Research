# Codex 科研执行任务：ROUND-027

# UniSparse剪枝损伤感知时序知识蒸馏最小机制及性能验证

## 1. 任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-027
- 实验编号：EXP-029
- 科研决策：DEC-027，IMPLEMENT + BOUNDED VALIDATION
- 理论版本：v0.3.3
- 科学问题：Q-016
- 科研假设：H-030、H-031、H-032
- 候选方法：PD-TKD，Pruning-Damage-Aware Temporal Knowledge Distillation
- 项目仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-026-compact-finetuning
- 本轮建议分支：research/round-027-pruning-damage-distillation
- 上轮训练代码SHA：7a72c4d574f9d6b69de5de76a110f3ddc9ded919
- 上轮修复后评价代码SHA：6e07d86e62485037f4265f49f0d975f4b28d3b14
- 上轮最终评价锁SHA：1141fae5bd8451e5d42a6e4bbb05b65510574cb0
- 上轮最终结果SHA：e42850b91fb57c02df41f5c77777c50357be73d6
- UniSparse官方固定SHA：b5b3da4fcd2e7abcd2c20c93d41cf63ef3201daa
- 依赖实验：EXP-018、EXP-024、EXP-027、EXP-028

P0：归档上一轮科学结论，修复跨阶段准入执行机制。

P0：验证实际逐时间步剪枝损伤及蒸馏数学公式。

P1：开展五组固定的分类读出微调，各5轮。

P1：判断真实剪枝损伤是否提供超出普通知识蒸馏的独立性能价值。

本轮是单一候选方法的有限筛选，不允许自动开展后续训练或额外方法搜索。

## 2. ROUND-026正式科研判定

EXP-028训练RUN-004及修复后的最终评价EXP-028-EVAL-001保留真实完成记录。

最终评价EVAL-001为COMPLETED / VALID。

提前启动的旧终点评价及原不完整评价锁保持INVALID。

保留RUN-001/002/003的AMP数学参考检查失败记录。

H-028更新为PARTIALLY_SUPPORTED：

普通全模型微调取得正向性能恢复，但未达到预注册至少1个百分点准确率提升的联合门槛。

H-029更新为INSUFFICIENT_EVIDENCE：

F1的CE低于F0，但准确率未获得预注册优势。

H-028/H-029的原整体门槛保持失败。

本轮没有观察到严重的任务恢复与计算稀疏性冲突，不能将该冲突描述为已证实的研究问题。

固定支持集最小二乘重构继续STOPPED。

UniSparse Controlled Baseline以及UniSparse＋L1＋标准微调强对照继续保留。

## 3. EXP-028真实实验依据

B0：

- Accuracy=94.222222%。
- CE=0.1727328626844618。
- FR=0.04957855784345042。
- SOP/sample=86009551.08977778。
- 参数量36720484。

P0：

- Accuracy=92.511111%。
- CE=0.27720849974950157。
- FR=0.04940481636399951。
- SOP/sample=74488295.33955556。
- 参数量11399524。

F0：

- Accuracy=93.288889%。
- CE=0.23999141862657336。
- FR=0.04940481636399951。
- SOP/sample=74488282.94244444。
- 参数量11399524。
- 性能恢复比例0.454545。
- 新增有效连接46。
- 失活有效连接62。
- 记录训练耗时69.73秒。

F1：

- Accuracy=93.244444%。
- CE=0.20323681349224515。
- FR=0.04984002698608942。
- SOP/sample=74861360.12177777。
- 参数量11399524。
- 性能恢复比例0.428571。
- 新增有效连接50213。
- 失活有效连接47097。
- 记录训练耗时145.30秒。

F1-P0准确率差：

+0.733333 pp，95% CI [+0.088889,+1.377778]。

F1-F0准确率差：

-0.044444 pp，95% CI [-0.622222,+0.555556]。

五轮微调不是标准训练已充分收敛的证据。

上述历史F0/F1只能在相同协议及明确训练预算下用于比较。

## 4. 科研动机及创新性限制

本轮研究问题：

固定高比例结构化剪枝造成的任务判别信息损伤是否具有时间异质性，以及利用精确损伤指导蒸馏是否比普通KD和均匀时间KD更有效。

已有研究：

1. Hinton、Vinyals、Dean，Distilling the Knowledge in a Neural Network，2015。

https://arxiv.org/abs/1503.02531

2. Knowledge from the original network: restore a better pruned network with knowledge distillation。

https://doi.org/10.1007/s40747-020-00248-y

3. Temporal Separation with Entropy Regularization for Knowledge Distillation in Spiking Neural Networks，CVPR 2025。

https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Temporal_Separation_with_Entropy_Regularization_for_Knowledge_Distillation_in_Spiking_CVPR_2025_paper.html

4. Efficient Logit-based Knowledge Distillation of Deep Spiking Neural Networks for Full-Range Timestep Deployment，ICML 2025。

https://proceedings.mlr.press/v267/yu25y.html

5. Not All Timesteps Matter Equally: Selective Alignment Knowledge Distillation for Spiking Neural Networks，IJCAI 2026。

https://ijcai.org/proceedings/2026/410

普通KD、逐时间步KD、非均匀时间权重均不是本项目首创。

本候选差异仅为利用固定结构删除造成的真实判别间隔损伤指导时序知识恢复。

不能将初步检索或实验精度提升描述为原创性证明。

## 5. 固定教师及初始剪枝模型

教师B0：

/home/hym/keyan1/experiments/EXP-026/models/B0.pth

SHA256：

cd6e151efe68ee77ef57ce474d0e83737321f271d4bace1ef8809013a66ae60d

初始剪枝模型P0：

/home/hym/keyan1/experiments/EXP-026/models/75-R1.pth

SHA256：

95073e1c9b3731a792fc72d596b3fa1c96b86989b328241b00bb663020c80958

交叉核查：

experiments/EXP-027/fixed_model_manifest.json

experiments/EXP-027/candidate_admission_lock.json

experiments/EXP-026/selected_neurons.json

B0和P0全部保持只读。

训练前后核对真实文件SHA256。

教师与固定P0参考始终处于eval模式。

教师和P0参考全部requires_grad=False。

不允许更新任何教师或参考模型参数、BN统计、阈值或其他持久化buffer。

五个训练学生均从原P0独立初始化。

不得从历史F0、F1或R3检查点继续训练。

## 6. 模型与代码审查

优先读取：

vendor/UniSparse/models/cifarnet.py

vendor/UniSparse/models/submodules/layers.py

experiments/ROUND-025/run.py

experiments/ROUND-026/run.py

experiments/ROUND-026/endpoint_repair.py

experiments/ROUND-026/gate.py

确认真实模型forward返回未经时间平均的分类输出：

Z.shape=[T,B,K]。

T=4，K=10。

教师最后隐层2048个神经元。

P0及学生最后隐层512个神经元。

fc1与fc2真实紧凑结构保持不变。

学生只允许训练fc2.weight和fc2.bias。

其余全部参数和持久化buffer冻结。

特征提取网络始终eval。

物理参数量固定11399524。

保持原LIF、CuPy后端、替代梯度及原有效权重参数化。

不修改vendor源码。

## 7. R127—R129精确时序剪枝损伤

捕获教师最后隐层二值脉冲：

s_teacher.shape=[T,B,2048,1,1]。

根据原实际fc2有效权重：

W2_eff.shape=[100,2048]。

类别读出权重：

A[k,i]=mean(W2_eff[10*k:10*k+10,i])。

类别偏置：

b[k]=mean(fc2.bias[10*k:10*k+10])。

R127：

z_teacher[t,n,k]=b[k]+sum_i A[k,i]*s_teacher[t,n,i]。

根据原锁定保留集合R及删除集合S：

R128：

E[t,n,k]=z_teacher[t,n,k]-z_P0[t,n,k]

理论上等于：

sum_{i in S} A[k,i]*s_teacher[t,n,i]。

对每个样本及时间步：

j[t,n]=argmax_{k!=y[n]} z_teacher[t,n,k]。

平局采用较小类别索引优先。

m_teacher[t,n]=z_teacher[t,n,y[n]]-z_teacher[t,n,j[t,n]]。

m_P0[t,n]=z_P0[t,n,y[n]]-z_P0[t,n,j[t,n]]。

注意P0必须使用教师选出的同一个竞争类别j。

R129：

m_teacher-m_P0=E_y-E_j。

该关系采用实际逐时间步logits验证。

严格FP32真实前向与FP64独立代数重构。

归一化最大误差容差沿用1e-5。

不得只验证时间平均后的logits。

不得使用原始潜在权重替代有效权重。

## 8. R130—R131剪枝损伤权重

R130：

d[t,n]=(m_teacher[t,n]>0)*relu(m_teacher[t,n]-m_P0[t,n])。

d.shape=[T,B]。

计算使用真实固定B0/P0输出。

不要使用正在训练的学生输出代替P0。

当教师真实类别间隔不为正时，该时间步损伤置0。

R131：

rho=0.5。

sum_d[n]=sum_t d[t,n]。

若sum_d[n]>0：

w[t,n]=(1-rho)+rho*T*d[t,n]/sum_d[n]。

若sum_d[n]=0：

全部w[t,n]=1。

必须使用安全分母避免未选分支发生0/0。

理论范围：

0.5<=w<=2.5。

mean_t(w)=1。

权重全部detach。

使用FP32或FP64稳定计算。

不得沿Batch维度进行归一化。

不得对全部样本使用统一时间权重。

不得搜索rho。

R130正部截断及教师正确条件属于理论启发式设计，不是已被证明的最优损伤指标。

## 9. R132—R134知识蒸馏公式

固定蒸馏温度：

tau_d=4。

固定蒸馏系数：

beta=0.5。

原任务损失：

CE=F.cross_entropy(z_student.mean(dim=0),labels)。

教师时间平均概率：

p_teacher_mean=softmax(z_teacher.mean(0)/tau_d,dim=-1)。

学生时间平均对数概率：

log_p_student_mean=log_softmax(z_student.mean(0)/tau_d,dim=-1)。

K1的KD：

L_KD_mean=tau_d²*mean_n KL(p_teacher_mean||p_student_mean)。

逐时间步：

p_teacher=softmax(z_teacher/tau_d,dim=-1)。

log_p_student=log_softmax(z_student/tau_d,dim=-1)。

KL_tn=sum_k p_teacher*(log_p_teacher-log_p_student)。

KL_tn.shape=[T,B]。

K2的KD：

L_KD_time=tau_d²*mean_{t,n}(KL_tn)。

R132：

L_PD=tau_d²*mean_{t,n}(w*KL_tn)。

R133：

L_total=CE+beta*L_KD。

R134：

w_shift[t,n]=w[(t+1)%T,n]。

K3使用：

L_shift=tau_d²*mean_{t,n}(w_shift*KL_tn)。

K3只错位权重，不改变教师logits、标签或学生时间步。

计算KL时显式使用FP32的softmax及log_softmax。

必要时关闭KL计算区域的autocast。

使用torch.nn.functional.kl_div时确认方向为teacher||student，reduction及类别求和正确。

不得反向传递至教师、固定P0或权重计算。

学生分类读出必须保留梯度。

CE沿用已经验收的原AMP任务公式。

不对CE与KD重新进行未经授权的梯度归一化。

## 10. 实验分组

K0：

CE。

标准分类读出微调。

K1：

CE+0.5*L_KD_mean。

K2：

CE+0.5*L_KD_time。

K3：

CE+0.5*L_shift。

M6：

CE+0.5*L_PD。

所有组：

- 相同P0初始化。
- 相同可训练参数。
- 相同数据和随机种子。
- 相同训练轮数。
- 相同学习率及调度。
- 相同原有效权重机制。
- 相同物理模型参数量。
- 不进行手动稀疏惩罚更新。

K0不需要教师前向。

K1、K2只需实际教师前向。

K3、M6需要固定教师与固定P0的输出。

报告实际训练计算开销差异。

若复用冻结特征或由教师脉冲精确重建P0输出，必须先验证该优化与真实P0前向在实际训练精度及增强输入下等价。

未验证时使用独立真实前向。

不能以离线固定D_grad logits替代每个增强训练样本的对应教师输出。

## 11. 数据与训练配置

沿用EXP-028：

CIFAR-10。

原训练集45000张。

原验证集4500张。

划分文件：

experiments/EXP-004/splits/indices.json

SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

继承原训练增强及归一化。

训练Batch Size=256。

workers=4。

drop_last=True。

每轮175个batch，实际44800张，丢弃200张。

五组各5个Epoch。

每组875次计划优化器尝试。

SGD。

momentum=0.9。

weight_decay=0。

初始lr=0.01。

CosineAnnealingLR：

T_max=5，eta_min=0。

每轮结束调用一次scheduler.step()。

AMP=True。

新建GradScaler。

seed=12450。

沿用EXP-028每轮样本顺序和worker种子。

教师、固定P0和当前学生必须使用完全相同的实际增强输入，不允许独立重新增强。

每次模型前向前后按真实需求reset各自神经元状态。

固定第5轮终点为正式比较对象。

不得根据验证集选择最佳Epoch。

禁止读取官方测试集。

不使用旧500张诊断池。

## 12. 训练前剪枝损伤诊断

使用固定512张D_grad样本。

索引沿用EXP-027，哈希：

93c4c834c3fd3c8b92a64caa88da8a671e27be23dc2df11999c8e5f3dfbe47d8

验证：

- 教师真实逐时间步logits重构。
- P0逐时间步删除损失重构。
- 保留神经元脉冲一致性。
- R129精确判别间隔差。
- d的有限性及非负性。
- w的边界及样本内平均为1。
- 正损伤时间步数量。
- 各时间步损伤均值、分位数。
- 非均匀权重样本比例。
- 权重错位后实际不同的元素数量。
- 原教师逐时间步准确率及时间平均准确率。
- 教师时间概率分布熵。
- P0逐时间步预测变化。

固定非退化进入条件：

至少10%的D_grad样本满足max_t(w)-min_t(w)>1e-6。

真实权重与循环错位权重必须存在非零差异。

该门槛仅用于避免开展没有有效时间权重干预的退化实验，不是科学假设已成立的证明。

若未达到，停止本轮正式训练。

不改变损伤定义、rho或诊断样本。

D_grad来自父模型训练集，不属于独立未见数据。

## 13. 数学及工程前置检查

至少完成：

1. 真正2+1不等尾批检查。
2. R127—R129人工及真实模型检查。
3. d全部为0的退化情况。
4. 单个时间步存在损伤。
5. 所有时间步损伤相同。
6. 教师错误时间步的处理。
7. 权重归一化及边界。
8. rho=0时与均匀时间KD一致。
9. K3错位前后每个样本权重多重集合相同。
10. 教师和学生logits相同时KD为0，允许合理浮点误差。
11. 所有KD公式与独立高精度参考一致。
12. 实际训练精度下损失及梯度有限。
13. K0实际AMP CE重复计算一致。
14. 五个候选的实际真实训练批次SGD更新。
15. 教师和P0无梯度、无持久化状态变化。
16. 学生全部非fc2参数及buffer无变化。
17. 原软阈值不变。
18. 手动稀疏惩罚调用次数为0。
19. 真实恢复检查点保存与加载正确。
20. 紧凑模型无原2048神经元结构残留。

FP64数学参考与真实AMP公式之间的舍入差异必须分别报告。

不能把FP16或混合精度误差直接当作数学实现错误。

在正式运行前确定并锁定合理的分精度测试标准。

若修改测试容差或计算参考，必须建立新Run ID并明确记录原因，不得覆盖原失败测试。

所有预检更新在临时模型执行。

正式训练必须从原P0重新初始化。

## 14. 强制拆分执行阶段

针对EXP-026和EXP-028重复出现的准入拦截偏差，本轮不允许单一连续主脚本在内部依次完成预检、训练和正式评价。

至少建立以下独立进程入口：

阶段A：preflight。

阶段B：train。

阶段C：endpoint_admission。

阶段D：evaluate。

阶段A完成后提交真实训练准入锁。

阶段B独立启动，并重新验证远程锁及实际模型身份。

阶段B只能训练、保存检查点和训练日志。

阶段B不得调用正式验证集评价函数，也不得自动启动阶段D。

阶段C独立读取实际终点恢复检查点和纯推理检查点，完成对象级检查。

阶段C必须生成并提交正式终点评价锁。

阶段D独立启动前再次核验远程评价锁、实际文件哈希和训练协议。

阶段D只有校验全部成功后才能创建正式验证集加载器。

所有阶段使用实际必需测试清单，而不是测试数量或硬编码True。

禁止通过主程序状态字段代替真实校验结果。

## 15. 准入负面测试

在实际执行入口测试：

- 缺少某个组的实际预检时拒绝训练。
- 教师或P0哈希变化时拒绝训练。
- 错误优化器参数范围时拒绝训练。
- 蒸馏系数或时间权重配置变化时拒绝训练。
- 训练准入锁未同步远程时拒绝训练。
- 缺少任一终点模型身份检查时拒绝评价。
- 缺少恢复模型与纯推理模型前向等价检查时拒绝评价。
- 终点检查点哈希变化时拒绝评价。
- 训练记录不完整时拒绝主要评价。
- 评价锁未同步远程时拒绝评价。

至少通过测试替身记录：当准入失败时，正式验证集加载器构造次数和正式评价函数调用次数均为0。

不能只测试独立validator返回False而不测试真实入口拦截。

任何异常退出都采用fail-closed原则。

## 16. 固定权重与状态控制

保留原UniSparse：

W_eff=sign(W_raw)*PseudoRelu(abs(W_raw)-threshold)。

软阈值全部固定。

教师、P0参考及学生冻结特征层不执行任何持久化状态更新。

学生只训练fc2权重和bias。

不主动增加有效连接。

允许原有效权重参数化下的连接自然重新激活或失活。

每轮完整检查全部非目标参数及持久化buffer。

不得仅抽查部分冻结参数。

记录实际学生输出与初始P0的脉冲一致性。

在实际固定诊断输入下，学生的保留神经元脉冲应逐元素一致。

若出现差异，首先检查模式设置、精度、神经元状态和冻结保护，不得直接继续正式评价。

## 17. 算法伪代码

对每个组独立执行：

加载固定P0作为student。

冻结除fc2.weight和fc2.bias之外的全部参数。

固定所有原软阈值。

创建相同配置的新SGD和scheduler。

按固定5轮训练数据：

    x,y=同一增强训练batch

    若为蒸馏组：
        no_grad:
            Z_teacher=teacher(x)

            若为K3或M6：
                Z_P0=fixed_P0(x)
                d=R130(Z_teacher,Z_P0,y)
                w=R131(d)

    Z_student=student(x)

    CE=CE(mean_t(Z_student),y)

    根据固定组别计算KD

    loss=CE+beta*KD

    K0的KD为0

    backward及optimizer step

    正确reset全部使用过的模型

    检查冻结参数、阈值及更新状态

    保存真实统计

每轮保存原子恢复检查点。

第5轮保存独立纯推理检查点。

训练进程退出。

之后独立执行终点准入与正式评价。

## 18. 训练过程记录

每轮保存：

- CE。
- KD。
- 总训练损失。
- 训练准确率。
- 学习率。
- 实际成功更新次数。
- AMP跳步。
- 实际采样顺序及丢弃索引。
- 各组增强随机控制。
- 非有限值检测。
- 冻结状态保护。
- 全部软阈值。
- 手动更新次数。
- 有效连接新增及失活数量。
- 实际参数量。
- GPU allocated/reserved峰值。
- 训练墙钟及CPU消耗。

K3和M6额外记录：

- 实际训练权重的均值、方差、最大值及最小值。
- 零损伤样本比例。
- 非均匀权重样本比例。
- 各时间步平均损伤及蒸馏损失。
- 错位权重与真实权重差异。

为控制仓库大小，可以保存聚合统计及固定诊断样本的逐样本数据，不必保存每个训练样本完整脉冲张量。

## 19. 正式终点评价

评价对象：

B0。

P0。

K0。

K1。

K2。

K3。

M6。

在实际终点评价锁通过后，统一评价固定4500张验证样本。

使用：

严格FP32。

关闭cuDNN TF32。

关闭matmul TF32。

关闭AMP。

Batch Size=64。

workers=4。

eval模式。

保留尾批。

正确reset神经元。

保存全部真实评价结果。

重新计算B0和P0，不复制历史数字。

K0与EXP-028 F0进行历史协议重建审计。

若K0存在差异，报告原因及实际样本预测差异。

不得修改K0结果以匹配历史F0。

如协议及初始化一致但实际重复存在非确定性差异，保留真实结果；若存在未解释的协议偏离，则停止相应机制归因。

## 20. 评价指标

主要指标：

Accuracy。

CE。

R126性能恢复比例。

真实物理参数量。

实际结构压缩率。

SOP_sample_weighted。

SOP_author。

FR_global。

逐层SOP及FR。

有效非零权重数及稀疏率。

相对P0的有效连接变化。

训练墙钟及资源成本。

纯推理模型文件大小。

逐样本预测、标签、正确性及CE。

额外记录逐时间步logits用于机制诊断：

Z.shape=[T,N,K]。

保存样本ID及正确的时间顺序。

计算教师与各学生的逐时间步KL、平均logits KL及判别间隔差。

以原P0损伤确定的时间权重作为固定分析依据。

不得根据这些终点诊断调整方法。

理论SOP不能解释为硬件实测能耗。

## 21. 预注册假设判定

H-030：

K1相对K0准确率提高至少0.3 pp。

H-031：

K2相对K1准确率提高至少0.3 pp。

H-032：

M6相对K2及K3准确率分别提高至少0.3 pp。

整体M6推进条件：

1. M6相对K0/K1/K2/K3准确率分别提高至少0.3个百分点。
2. B0与M6准确率差不超过0.5个百分点。
3. M6验证CE不高于K2。
4. M6物理参数量固定11399524。
5. M6全网络SOP不超过P0的101%。
6. 全部训练及正式评价准入通过。
7. 组间训练协议及数据控制有效。

所有数值门槛为探索性资源推进标准，不是统计显著性定义。

H-030、H-031、H-032分别判定，不强制前一个假设成立才能报告后一个比较。

若M6优于K0但未优于普通KD：

不能宣称损伤感知设计具有独立性能价值。

若M6未优于K3：

不能宣称真实损伤与对应时间步的匹配是改善来源。

若M6未达到整体门槛：

不自动修改rho、beta、tau_d、剪枝率、训练轮数或蒸馏目标。

科学状态由理论研究员审查决定。

## 22. 配对统计

固定：

2000次配对Bootstrap。

seed=20261128。

numpy.default_rng PCG64。

百分位95%区间。

主要比较：

K1-K0。

K2-K1。

M6-K0。

M6-K1。

M6-K2。

M6-K3。

M6-B0。

辅助比较：

各组相对P0的任务恢复。

报告准确率百分点差、CE差、置信区间。

保存参考错候选对、参考对候选错、预测不一致数量。

逐样本SOP缺失时对应置信区间标记NA。

多项比较的95%区间为未校正探索性区间。

不能将单训练种子结果及样本Bootstrap描述为跨种子稳定性。

## 23. 资源及停止条件

需要用户新授权。

建议：

单张符合实验室共享管理规定的RTX4090。

GPU总占用上限30分钟。

CPU累计计算上限90分钟。

正式训练总量最多25个Epoch。

所有前置检查、资源基准、失败重试、训练和正式评价计入预算。

正式训练前进行必要短程测量，确认预算可行。

不得擅自降低Batch Size或增加资源。

出现以下任一情况时停止对应阶段：

- 原模型哈希不一致。
- 数据隔离失败。
- 真实逐时间步损伤公式检查失败。
- 时间权重完全退化或不满足预设进入条件。
- 教师或P0参数、BN及持久化buffer变化。
- 学生非目标参数变化。
- 软阈值变化。
- 手动稀疏更新。
- 物理参数量变化。
- 缺少真实候选准入。
- 缺少正式远程锁。
- 非有限损失、梯度或参数。
- OOM。
- 资源预算不足或超限。

不干扰其他用户GPU进程。

不因会话中断重置资源预算。

未完成5轮的组不得纳入固定终点主要比较。

## 24. 科研状态更新

新增：

research/ROUND-027-TASK.md

更新：

research/THEORY.md，追加v0.3.3及R127—R134。

research/HYPOTHESES.md，记录H-028/H-029正式判定，新增H-030—H-032。

research/RESEARCH_STATE.md。

research/EXPERIMENT_PLAN.md。

research/EXPERIMENT_REGISTRY.csv。

research/DECISION_LOG.md，追加DEC-027。

research/LITERATURE.md，追加上述真实文献及重复风险。

AGENTS.md追加四阶段独立执行入口及真实入口负面拒绝测试规范。

不修改任何历史失败记录。

不恢复已停止的M5、SEAWP、MP-EFR、TP-EFR、ARTG及固定支持集重构路线。

## 25. 交付文件

新增：

experiments/ROUND-027/

experiments/EXP-029/

至少保存：

- TASK.md
- config.json
- environment.txt
- code_version.json
- commands.sh
- status.json
- parent_checkpoint_audit.json
- teacher_protection_audit.json
- architecture_audit.json
- data_split_audit.json
- formula_tests.json
- temporal_readout_audit.json
- damage_diagnostics.csv
- damage_summary.json
- weight_diagnostics.csv
- optimizer_parameter_audit.json
- preflight_tests.json
- admission_coverage_matrix.csv
- admission_negative_tests.json
- training_admission_lock.json
- training_admission_lock_commit.json
- epoch_metrics.csv
- training_sample_audit.csv
- distillation_metrics.csv
- frozen_parameter_audit.json
- threshold_audit.csv
- topology_transition.csv
- checkpoint_manifest.json
- evaluation_admission_lock.json
- evaluation_lock_commit.json
- evaluation_entry_audit.json
- validation_summary.csv
- validation_per_sample.csv
- temporal_logits_manifest.json
- temporal_distillation_diagnostics.csv
- layer_sops.csv
- paired_bootstrap.json
- hypothesis_gate_results.json
- resource_usage.csv
- REPORT.md
- limitations.md
- run.log
- figures/

图表至少包括：

1. 真实剪枝损伤的时间步分布。
2. 实际损伤权重及错位对照。
3. 五组训练CE及KD曲线。
4. 固定压缩预算下的准确率及CE。
5. 各组SOP和参数压缩。
6. 主要配对准确率差及置信区间。

未产生的文件或指标标记NA及真实原因。

大型模型、完整logits数据可保留服务器，记录真实路径、shape、dtype、样本顺序及SHA256。

不得生成虚构结果。

## 26. GitHub与最终交付

从ROUND-026最终结果SHA创建独立分支。

开始前读取AGENTS.md及当前科研状态。

提交固定任务、代码、配置和判断门槛。

训练前提交真实训练准入锁。

正式评价前提交独立终点准入锁。

所有结果、失败记录、实际执行版本及科研报告同步GitHub。

不修改main。

不force push。

不覆盖历史实验。

最终返回：

- GitHub实验分支链接。
- 实际预检、训练、终点审计及评价代码SHA。
- 训练准入锁SHA。
- 正式评价准入锁SHA。
- 最终结果提交SHA。
- EXP-029执行状态及有效性。
- R127—R134实际公式验证结果。
- 剪枝损伤非退化进入条件。
- 五组实际完成Epoch。
- 七个模型的Accuracy、CE、FR、SOP及参数量。
- K0历史F0重建情况。
- M6相对K0/K1/K2/K3的配对统计。
- H-030/H-031/H-032及整体门槛结果。
- 教师、P0及冻结状态审计。
- 实际CPU/GPU资源消耗。
- 全部异常及局限。
- 实验报告和原始数据路径。
- 最终模型路径及SHA256。

本轮只执行一次固定候选方法筛选。

不进行超参数搜索，不开展额外训练，不自动进入下一轮。

获得用户授权后执行。