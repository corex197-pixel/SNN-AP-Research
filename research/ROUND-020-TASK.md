# Codex 科研执行任务：ROUND-020

# UniSparse任务梯度冲突诊断与任务保护型有效扇出正则化验证

## 1. 任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-020
- 科研阶段：阶段1—3，候选创新机制修正与最小验证
- 科研决策：DEC-020，REFINE
- 理论版本：v0.2.6
- 科学问题：Q-010
- 科研假设：H-016、H-017、H-018、H-019、新增H-020、H-021
- 实验编号：EXP-022
- 候选方法：TP-EFR，Task-Protected Effective Fan-out Regularization
- 项目仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-019-matched-budget
- 本轮建议分支：research/round-020-task-protected-fanout
- 上轮执行代码SHA：912655152ed612e8542e79d2044336f1e16c2c3b
- 上轮最终结果SHA：f26ae97b86363930c88f42e9e9d97febb9d40dce
- UniSparse官方固定SHA：b5b3da4fcd2e7abcd2c20c93d41cf63ef3201daa
- 依赖实验：EXP-018、EXP-019、EXP-020、EXP-021

优先级：

P0：真实任务梯度及活动惩罚梯度的冲突诊断。

P1：任务保护投影正确性与实际效果。

P1：2×2因子设计的同SOP预算性能比较。

本轮为当前MP-EFR方向的一次有限修正实验。

禁止SGD微调、完整训练、额外超参数搜索、修改神经元动力学或使用官方测试集。

## 2. ROUND-019正式科研判定

EXP-021为COMPLETED。

校准及验证预算匹配均通过。

整体推进门槛未通过。

固定验证集结果：

B0：

- Accuracy=0.9424444444444444。
- CE=0.17519400440322028。
- SOP=86008931.65022223。

G0：

- q=0.04546005512832126。
- Accuracy=0.9433333333333334。
- CE=0.18290079932742648。
- SOP=82163161.20311111。
- 参数更新L2=1.3139499165937578。

G1：

- q=0.04。
- Accuracy=0.9428888888888889。
- CE=0.17795569733778635。
- SOP=82161247.10466667。
- 参数更新L2=1.1561357713785758。

G2：

- q=0.04662144833450371。
- Accuracy=0.9375555555555556。
- CE=0.1916092533270518。
- SOP=82197501.672。
- 参数更新L2=1.3475181359863546。

G1-G0准确率差：

-0.044444个百分点。

95%配对Bootstrap CI：

[-0.488889, 0.422778]个百分点。

G1-G2准确率差：

+0.533333个百分点。

95% CI：

[0.066667, 1.022222]个百分点。

G1相对G0的CE差：

-0.004945099167752763。

95% CI：

[-0.013591332584099542, 0.004032630454181975]。

G1达到近似相同SOP预算所需参数更新范数相对G0减少约12.01%，相对G2减少约14.20%。

该结果支持局部计算稀疏化效率，但不能替代任务性能优势。

EXP-021原推进门槛保持失败。

不事后修改原协议。

## 3. 科学假设及研究问题

H-016保持SUPPORTED及原适用范围。

H-018保持PARTIALLY_SUPPORTED及原负面证据。

H-019保持SUPPORTED，仅限当前固定模型的局部计算效率。

H-017保持INSUFFICIENT_EVIDENCE，并补充：

EXP-021预算匹配成功后，G1未达到相对G0的预注册准确率优势门槛。不能再将证据不足归因于缺少公平对照。

H-020：

UNTESTED。

在当前U2第300轮固定模型中，MP-EFR的部分活动惩罚梯度可能与全网络任务损失的替代梯度发生冲突。

可观测预测：

至少一个目标BN模块满足：

dot(t_l,g1_l)<0。

若两个目标模块均不存在可分辨的负内积，则本轮任务保护动机缺乏直接支持。

H-021：

UNTESTED。

在有效权重掩码固定、相近实际SOP预算条件下，任务保护型有效扇出方向P1可能比G0、G1及P0具有更好的任务性能。

不能提前认定梯度冲突是EXP-021失败的真实原因。

## 4. 固定父模型

使用：

experiments/EXP-018/runs/EXP-018-CONTINUE-001/U2/epoch-300.pth

SHA256：

19b1312cbaae49ad95a7d001270c4357d7ea23b2584d19698b0084b3da4b094f

运行前后核对父检查点哈希。

保持只读。

不加载best_validation代替固定300轮模型。

不恢复optimizer进行训练。

每次参数干预均从同一父模型开始。

## 5. 固定目标模块

仅允许干预：

路径L1：

static_conv.norm -> static_conv.activation -> conv1.conv

路径L2：

conv1.norm -> conv1.activation -> conv2

每个目标BN包括gamma和beta。

其他参数、卷积权重、权重阈值、有效掩码及BN运行统计保持不变。

沿用EXP-020/021已经验证的直接消费关系。

不增加目标层。

## 6. 原始活动梯度

复用R74—R80。

r[c]=mean(s,dim=(0,1,3,4))

s.shape=[T,B,C,H,W]

d[c]=sum(M[:,c,:,:])

d0=C_out*k_h*k_w

M为后继卷积实际有效二值掩码。

a[c]=(1-rho)+rho*d[c]/mean(d)

rho固定0.25。

mean(d)=0时回退a=1。

G0：

P0_original=d0*sum(r[c])

G1：

P1_original=d0*sum(a[c]*r[c])

活动梯度分别为：

g0_l=gradient(P0_original,phi_l)

g1_l=gradient(P1_original,phi_l)

phi_l=concat(gamma_l,beta_l)

继续使用冻结BN运行统计的局部eval重放、原CuPy LIF和ATan替代梯度。

优先读取并核验：

experiments/EXP-021/gradient_vectors.csv

该文件SHA256：

19426f0b80dd5bcce4c09a4cc911a6cb3b5279b52decf21b44cb6834cc1a591c

确认实际文件格式、参数顺序、raw/matched记录及数值精度。

若无法安全复用，则按原协议重新计算梯度。

不能直接复用未经核验的向量。

## 7. 固定数据及环境

继续使用EXP-020和EXP-021的实际样本索引。

D_grad：512张。

D_cal：512张。

验证集：4500张。

固定划分：

experiments/EXP-004/splits/indices.json

SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

D_grad索引哈希：

93c4c834c3fd3c8b92a64caa88da8a671e27be23dc2df11999c8e5f3dfbe47d8

D_cal索引哈希：

b671cbd803b13cc76b02f5db4bad311ef9492276622051df01af56a931301baa

沿用原序列化方式。

不重新采样，不使用500张旧诊断池，不读取官方test_batch。

D_grad/D_cal曾参与父模型训练。

验证集已经参与历史工作点选择及多轮开发比较，因此本轮仍为探索性实验，不是独立确认实验。

固定配置：

- CifarNet。
- base_channels=256。
- T=4。
- 官方CuPy LIF。
- 原UniSparse专用环境。
- FP32。
- Batch Size=64。
- workers=4。
- seed=20261128。
- 确定性评价预处理。

无随机数据增强，无optimizer更新。

## 8. R83：全网络任务梯度

定义：

z=mean_t(model_output)

L_task=CrossEntropy(z,label)

t_l=gradient(L_task,phi_l)

注意：

任务梯度必须通过完整网络任务反向传播计算，不能以局部活动代理梯度替代。

保持全部BN使用eval运行统计。

只有两个目标BN的gamma和beta需要梯度。

其余参数requires_grad=False。

但是禁止使用no_grad或detach截断目标BN至最终任务输出的必要计算路径。

由于SpikingJelly的eval神经元可能采用不提供替代梯度的推理路径，允许仅将实际LIF节点设置为能够计算原ATan替代梯度的训练路径。

必须保持BN、Dropout及其他影响真实前向行为的模块为eval状态。

逐批检查：

启用替代梯度计算前后的真实模型logits与脉冲状态一致。

若必要前向一致性无法建立，停止任务梯度诊断，不自行替换神经元后端。

每个Batch：

1. 正确reset所有神经元状态。
2. 真实完整网络前向。
3. 计算实际任务交叉熵。
4. 使用autograd.grad取得两个目标BN的gamma/beta梯度。
5. 按Batch实际样本数加权累计。
6. detach已获得梯度。
7. 释放计算图。
8. reset网络状态。

最终：

t_l=sum_b(n_b*t_l,b)/sum_b(n_b)

仅使用D_grad的512张样本估计主要任务梯度。

保存完整任务梯度向量及逐Batch诊断。

不调用optimizer.step。

不改变任何模型状态。

## 9. R84—R85：梯度冲突诊断

对原始G0和G1活动梯度分别计算：

dot_l=dot(t_l,g_l)

cos_l=dot(t_l,g_l)/(norm(t_l)*norm(g_l))

若任一向量范数为0，则cos标记NA。

使用float64进行点积及范数计算。

一阶代理预测：

Delta_L_task≈-eta*dot(t_l,g_l)

该表达式基于局部替代梯度近似，不是硬阈值SNN真实损失变化定理。

另外记录：

- 每个目标模块任务梯度范数。
- 活动梯度范数。
- 梯度内积。
- 梯度余弦相似度。
- 冲突分量范数占比。
- 每个通道的任务与活动梯度内积。
- 负内积通道比例。
- G0和G1冲突程度差异。
- D_grad前256张和后256张任务梯度方向一致性。

两半数据只用于稳定性诊断，不根据其结果选择新的任务梯度。

若G1所有目标模块均满足非负内积，或负余弦绝对值不超过1e-6的数值诊断容差：

记录未观察到可分辨冲突。

H-020方向门槛失败。

不执行不具有实质区别的P1重复性能实验。

不自动搜索其他样本、其他层或其他任务损失以制造冲突。

## 10. R86—R87：任务保护投影

对G0或G1的原始活动梯度g_l，求解：

min_v 0.5*norm(v-g_l)^2

subject to dot(t_l,v)>=0

闭式解：

v_l=g_l-min(0,dot(t_l,g_l))/norm(t_l)^2*t_l

建议数值稳定实现：

若norm(t_l)>0：

u_l=t_l/norm(t_l)

v_l=g_l-min(0,dot(u_l,g_l))*u_l

否则：

v_l=g_l

全部投影使用float64。

不对真实范数任意添加epsilon改变投影含义。

P0：

对G0原始活动梯度执行上述投影。

P1：

对G1原始活动梯度执行上述投影。

只保护任务梯度方向，不同时修改任务梯度自身。

不把任务梯度与稀疏梯度直接相加。

不执行优化器训练。

投影后检查：

dot(t_l,v_l)>=0，允许预先规定的浮点误差。

建议在非零范数情况下使用归一化余弦下界-1e-8检查。

保存投影前后梯度、投影移除分量及角度变化。

## 11. R88—R89：范数匹配与参数干预

沿用R79的每层范数匹配。

对G0、G1、P0、P1分别计算：

direction_l=norm(g0_l)*v_l/norm(v_l)

对于G0、G1：

v_l为其未投影的原始活动梯度。

若norm(g0_l)=0：

全部对应层更新为0。

若norm(g0_l)>0且：

norm(v_l)/norm(g0_l)<=1e-8

则记录投影退化，不强制放大数值噪声。

对应候选无法完成原等范数协议时标记无效，不擅自使用其他方向。

整体干预尺度：

eta(q)=q*norm(phi)/norm(g0)

其中phi连接两个目标BN的父模型gamma和beta。

g0连接两个目标BN的原始G0梯度。

最终：

phi_new_l=phi_parent_l-eta(q)*direction_l

每次从原父模型恢复，不累积应用。

实际更新范数、权重掩码、BN运行统计及非目标参数均需审计。

## 12. 实验分组

B0：

原父模型，无干预。

G0：

均匀活动惩罚，使用EXP-021锁定q=0.04546005512832126。

G1：

MP-EFR，rho=0.25，使用EXP-021锁定q=0.04。

P0：

均匀活动惩罚+任务保护投影。

P1：

MP-EFR+任务保护投影。

P0/P1的q根据校准集SOP匹配确定。

本轮不增加G2随机置换实验。

EXP-021已有G2证据完整保留。

G0/G1/P0/P1构成2×2因子设计。

不允许省略P0。

否则无法区分任务保护本身与有效扇出信息的贡献。

## 13. EXP-022A：实际冲突检查

测量第9节全部梯度指标。

H-020主要方向条件：

G1至少一个目标模块的梯度余弦小于-1e-6，且任务梯度、活动梯度均有效。

若未达到：

不进入正式P1性能评价。

记录H-020当前条件下不获得支持。

不得因为EXP-021性能未通过而直接宣称存在梯度冲突。

若达到：

继续检查投影数学正确性、实际前向一致性及非退化条件。

## 14. EXP-022B：相同干预尺度机制诊断

仅在H-020方向条件及正确性检查通过后执行。

固定主要诊断尺度：

q=0.02。

使用D_cal真实完整网络前向。

评价B0/G0/G1/P0/P1。

记录：

- 实际参数更新范数。
- CE及Accuracy。
- 全网络SOPs。
- 两个目标消费者合计SOPs。
- 全局及目标层FR。
- 投影前后任务一阶代理预测。
- 实际任务损失变化。

主要机制观察：

P1相对G1是否减少实际任务CE损失。

P0相对G0是否出现类似改善。

记录预测方向与真实损失变化是否一致。

即使代理投影满足数学约束，也不能自动宣称真实任务损失受到了保护。

同尺度下各组SOP可能不同，因此不能以本阶段准确率比较替代同预算性能评价。

## 15. EXP-022C：固定预算匹配

G0/G1保留EXP-021实际锁定q，不重新选择。

首先在D_cal进行必要的B0/G0/G1历史结果重建。

实际G1锚点：

q=0.04。

历史D_cal：

C_B0=86348901.18554688。

C_G1=82539922.0390625。

历史降幅约4.41115%。

本轮重新核验后以实际G1成本作为唯一目标。

不根据验证集结果改变锚点。

P0/P1均使用纯SOP成本选择q。

定义：

R_m(q)=1-C_m(q)/C_B0

R_star=R_G1(0.04)

要求：

abs(R_m(q)-R_star)<=0.001

且R_m(q)处于4%—6%区间。

即与G1校准降幅差不超过0.1个百分点。

固定有限匹配协议：

1. 优先使用已测q=0及q=0.02结果。
2. 测量q=0.04。
3. 必要时测量q=0.08。
4. 若仍未包围目标，允许测量q=0.16。
5. 一旦获得有效包围区间，依据R82线性插值得到一个候选。
6. 如该点不匹配，每组最多追加4次区间中点。
7. 不扩大最大q=0.16。
8. 不根据Accuracy或CE选择q。

若多个已测点均满足匹配条件，选择SOP误差最小者，平局选择较小q。

若出现非单调响应，保留全部实际观测，不假定二分法具有严格保证。

若P0由于无梯度冲突而与G0方向完全一致，可以直接使用G0的已锁定q，但必须完成实际方向及输出一致性检查。

若P1投影方向与G1完全相同或实质退化，记录本轮方法无法产生有效干预差异。

不进行额外超参数搜索。

## 16. 锁定与验证

在正式验证之前保存并提交：

selection_lock.json

包括：

- 父模型哈希。
- 固定样本索引哈希。
- 任务梯度及活动梯度哈希。
- 投影前后方向哈希。
- G0/G1历史固定q。
- P0/P1实际匹配q。
- 所有候选SOP及预算误差。
- 预注册推进条件。

验证开始后禁止重新选择q。

在全部主要比较组具备有效匹配候选时，使用原4500张验证样本评价B0/G0/G1/P0/P1。

FP32及Batch64协议不变。

检查B0/G0/G1与EXP-021历史输出的一致性。

若部分组未能匹配，应明确标记公平比较证据不足。

不得伪造完整实验或比较不同成本条件下的准确率后宣称同预算优势。

## 17. 评价指标

每组记录：

- Accuracy。
- CE。
- FR_global。
- 逐层消费者输入FR。
- SOP_sample_weighted。
- SOP_author。
- 逐层SOP。
- 相对B0的SOP降幅。
- 全部有效权重稀疏率。
- 有效非零权重数量。
- 实际总参数更新范数。
- 每层实际参数更新范数。
- R81局部优化效率。

保留7层原SOP监控范围。

验证尾批不得丢弃。

按实际样本数量加权。

理论SOP不等于硬件能耗或真实推理加速。

保存逐样本预测、标签、正确性及CE。

若可以在不改变原前向语义的情况下采集逐样本SOP，也一并保存；否则标记缺失及原因。

## 18. 配对统计与因子分析

固定2000次配对Bootstrap。

seed=20261128。

使用numpy.default_rng PCG64及百分位区间。

主要比较：

P1-G0。

P1-G1。

P1-P0。

次要比较：

P0-G0。

G1-G0。

P1-B0。

报告准确率百分点差、CE差、置信区间及配对错误变化。

另外计算固定实际SOP预算下的探索性2×2交互项：

I_Acc=(Acc_P1-Acc_G1)-(Acc_P0-Acc_G0)

I_CE=(CE_P1-CE_G1)-(CE_P0-CE_G0)

对同一验证样本四组结果进行联合配对重采样。

报告交互项及95%区间。

该交互项描述当前匹配预算下的结果差异，不直接证明训练因果机制。

由于验证集已重复用于开发，禁止将本轮区间包装为独立确认性检验。

## 19. 预注册推进与终止规则

本轮P1建议进入下一阶段的必要条件：

1. H-020真实梯度冲突方向条件通过。
2. 所有公式及状态检查通过。
3. 投影方向非退化且数学约束通过。
4. P0/P1校准预算匹配误差均不超过0.1个百分点。
5. P1验证SOP降幅处于4%—6%。
6. P1与G0/G1/P0验证SOP降幅差分别不超过1个百分点。
7. P1相对B0准确率下降不超过0.5个百分点。
8. P1相对G0准确率提高至少0.3个百分点。
9. P1相对G1准确率提高至少0.3个百分点。
10. P1相对P0准确率提高至少0.3个百分点。
11. 权重掩码及非目标模型状态保持不变。

以上是探索性推进门槛，不是统计显著性定义。

同时报告第14节实际CE保护结果，判断任务保护的理论预测是否得到真实前向支持。

若P1仅优于G1，但未优于P0：

不能宣称有效连接感知具有独立性能贡献。

若P1仅降低CE，但未达到准确率推进门槛：

保留结果，不修改主要判据。

若公平实验有效而推进门槛失败：

不推荐继续当前MP-EFR/TP-EFR正式训练主线。

不得自动调整rho、保护强度、目标层或新增蒸馏模块。

若实验因工程错误或预算不匹配而无效，应明确区分科学结论不足和方法性能失败。

若全部通过，仅推荐下一轮短程训练审批，不自动训练。

## 20. 最低正确性测试

至少检查：

- 非冲突梯度保持原方向。
- 负内积情况下投影结果满足正交边界。
- 任务梯度为零时回退原方向。
- 原梯度为零。
- 完全反向梯度导致投影退化。
- 近零梯度数值稳定性。
- 投影是R86半空间最小距离解。
- 投影后正缩放不改变任务内积符号。
- 真实任务梯度计算前向与eval一致。
- 真实R87及R88成立。
- q=0恢复父模型。
- 全部非目标参数与父模型一致。
- 全部有效掩码不变。
- BN运行统计不变。
- 样本配对及隔离正确。
- 实际SOP监控与独立计数一致。

不重复执行与本任务无关的全部历史工程测试。

## 21. 代码修改范围

读取并复用：

experiments/ROUND-018/math_core.py

experiments/ROUND-019/core.py

experiments/ROUND-019/run.py

以及EXP-021已交付数据。

新增独立：

experiments/ROUND-020/

experiments/EXP-022/

不得直接运行会覆盖历史EXP-020/021输出的入口。

保持vendor/UniSparse官方代码不变。

只新增任务梯度测量、投影、范数匹配、预算评价及必要测试。

不修改原稀疏算法或神经元动力学。

## 22. 学术创新性限制

R87的冲突梯度投影与已有PCGrad机制实质相同。

参考：

Yu et al., Gradient Surgery for Multi-Task Learning, NeurIPS 2020.

https://arxiv.org/abs/2001.06782

不得将梯度投影、正交约束或该闭式解描述为学术首创。

本轮研究价值在于验证高稀疏SNN中有效连接代价感知与任务保护结合的实际作用。

是否构成独立论文贡献仍取决于后续实验和文献查重。

更新LITERATURE.md，记录PCGrad及与当前候选方法的实质联系。

## 23. 资源预算

需要用户新授权。

建议：

- 单张符合实验室共享规定的RTX4090。
- GPU总占用上限15分钟。
- CPU累计计算上限30分钟。
- SGD训练Epoch=0。
- 不使用第二张GPU。

所有任务梯度反向传播、前置检查、匹配、验证及重试均计入预算。

实际峰值显存需记录。

本轮增加完整网络任务反向，显存需求可能高于EXP-021。

启动前检查实时GPU可用显存。

若Batch64出现OOM，停止受影响运行并记录。

不得未经确认改变正式Batch Size或精度口径。

发生以下情况停止相关任务：

- 父检查点哈希不一致。
- 任务梯度路径无法正确建立。
- 前向一致性失败。
- 投影退化。
- 非有限数值。
- 数据隔离失败。
- 权重掩码或其他状态改变。
- OOM或资源超限。
- 不能建立公平预算对照。

不因结果不理想自动扩大研究范围。

## 24. 科研状态更新

新增：

research/ROUND-020-TASK.md

更新：

- THEORY.md：追加v0.2.6及R83—R89。
- HYPOTHESES.md：记录H-017新证据，新增H-020/H-021。
- RESEARCH_STATE.md。
- EXPERIMENT_PLAN.md。
- EXPERIMENT_REGISTRY.csv。
- DECISION_LOG.md：追加DEC-020。
- LITERATURE.md。

EXP-020和EXP-021保持COMPLETED。

两轮原推进门槛失败记录完整保留。

H-019保持限定SUPPORTED。

不得将工程公式测试成功视为H-021成立。

## 25. 结果交付

在EXP-022独立目录保存：

- TASK.md
- config.json
- environment.txt
- code_version.json
- commands.sh
- status.json
- parent_checkpoint_audit.json
- historical_reconstruction_audit.json
- data_split_audit.json
- task_gradient_vectors.csv
- activity_gradient_vectors.csv
- gradient_conflict_statistics.csv
- channel_conflict_statistics.csv
- gradient_stability.json
- projected_gradient_vectors.csv
- projection_audit.json
- forward_equivalence_audit.json
- formula_tests.json
- mechanism_same_q.csv
- calibration_response.csv
- budget_matching_trace.csv
- selection_lock.json
- parameter_integrity_audit.json
- update_norm_audit.csv
- validation_summary.csv
- validation_per_sample.csv
- layer_sops.csv
- paired_bootstrap.json
- factorial_interaction.json
- hypothesis_gate_results.json
- resource_usage.csv
- REPORT.md
- limitations.md
- run.log
- figures/

图表包括：

1. 两个目标模块梯度冲突程度。
2. 投影前后任务梯度内积及余弦。
3. 实际任务损失变化与代理预测。
4. 同干预尺度SOP与CE。
5. 四组同预算性能比较。
6. 主要配对准确率差及置信区间。
7. 有效扇出与任务保护的因子分析。

若实验因真实门槛或资源限制提前终止，明确记录未执行文件及原因。

不生成虚构数据。

所有Run ID独立，失败记录不得覆盖。

## 26. GitHub交付及最终反馈

从上轮最终结果SHA建立独立分支。

读取并遵守AGENTS.md。

正式运行前提交任务、配置、判定规则和代码。

验证前提交实际预算匹配锁。

最终同步真实结果及科研状态。

不修改main，不force push，不覆盖历史数据。

最终返回：

- GitHub实验分支链接。
- 实际执行代码SHA。
- 最终结果提交SHA。
- EXP-022执行状态。
- 真实任务梯度与活动梯度冲突结果。
- G0/G1各目标层梯度内积与余弦。
- 投影实际修改程度。
- q=0.02同尺度机制结果。
- P0/P1实际匹配q。
- 所有组真实Accuracy、CE、FR和SOP。
- 配对统计及2×2交互结果。
- 全部推进门槛结果。
- 实际GPU/CPU资源。
- 异常、限制与未完成项目。
- REPORT及原始数据路径。

本轮是当前MP-EFR方向的有限修正验证。

不应为了维持该研究方向而无限增加模块。

获得用户资源授权后执行。