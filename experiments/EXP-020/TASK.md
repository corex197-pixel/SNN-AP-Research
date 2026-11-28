# Codex 科研执行任务：ROUND-018

# UniSparse 有效扇出感知活动惩罚的最小机制与性能验证

## 1. 任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-018
- 科研阶段：阶段1—3，真实理论机制验证与最小创新方法先导
- 科研决策：DEC-018，IMPLEMENT + CONTINUE
- 理论版本：v0.2.4
- 科学问题：Q-010，连接稀疏性与活动稀疏性的耦合优化
- 科研假设：H-016、H-017、H-018、新增H-019
- 实验编号：EXP-020
- 候选方法：MP-EFR，Mean-Preserving Effective Fan-out Regularization
- 中文名称：均值保持的有效扇出感知活动正则化
- 项目仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-017-unisparse-baseline
- 本轮建议分支：research/round-018-fanout-regularization
- 上轮实际执行SHA：3c510be030909b9459c1a2d574974c07e9f88830
- 上轮最终结果SHA：6ead381fb25eee81ee0e15d6ab3edd8ffc91e8b1
- UniSparse官方固定SHA：b5b3da4fcd2e7abcd2c20c93d41cf63ef3201daa
- 依赖实验：EXP-016、EXP-017、EXP-018、EXP-019

优先级：

P0：模型、数据、理论公式及实际计算图核查。

P1：原活动方向、有效扇出活动方向及置换对照的机制验证。

P1：相同SOP预算下的真实验证集性能筛选。

本轮禁止完整训练、SGD微调、从头训练、额外超参数搜索及使用官方测试集。

## 2. ROUND-017正式科学结论

EXP-018和EXP-019均为COMPLETED。

U0/U2各完成300轮。

第300轮固定验证结果：

U0：

- Accuracy=0.944。
- CE=0.1821459575229221。
- FR=0.05117369012266473。
- SOP_sample_weighted=401025556.71466666。
- 全部SparseConv2d权重稀疏率=8.170988496924991e-08。

U2：

- Accuracy=0.9424444444444444。
- CE=0.17519400702582466。
- FR=0.04957846704501978。
- SOP_sample_weighted=86008931.7048889。
- 全部SparseConv2d权重稀疏率=0.1447020508963247。
- 受惩罚层权重稀疏率=0.14472929757238895。
- 最低受惩罚层连接密度=0.05173068576388889。

U2相对U0准确率下降0.15556个百分点，理论SOPs下降约78.55%。

上述结果为单随机种子、训练来源验证集上的Controlled UniSparse Baseline，不是论文表格精确复现，不是硬件实测能耗。

正式论文非零lambda仍为UNKNOWN。

原监督进程退出原因未知，重新启动后完成交付，训练未中断。

保留原审计局限，不重写历史记录。

## 3. 科研假设状态更新

H-016：SUPPORTED，严格限定于当前模型和工作点中的名义扇出与有效扇出失配现象。

EXP-017及EXP-018真实掩码统计支持该描述性机制。

其任务性能影响及真实训练优化收益仍待验证。

H-018：PARTIALLY_SUPPORTED。

第60轮预注册主要评价点：

Delta_C_W=-215420519.17533332。

Delta_C_S=7578117.980666652。

Delta_C_S/C00=0.024091934798092594。

方向条件通过，但原预注册5%探索性幅度门槛未达到。

第300轮：

Delta_C_W=-313951463.0611111。

Delta_C_S=-1065162.6684444398。

Delta_C_S/C00=-0.00265609672381235。

因此不再以活动重分布持续抵消大量连接剪枝收益作为论文主线。

保留全部正负结果，不降低原门槛。

H-017：UNTESTED。

有效连接感知活动惩罚是否改善同SOP预算下的任务性能仍未知。

新增H-019：UNTESTED。

在固定权重和BN运行统计、相同局部参数更新范数的条件下，MP-EFR可能比原名义扇出活动惩罚获得更大的目标消费者SOP下降。

本轮验证H-019的局部方向预测，并对H-017进行探索性性能筛选。

## 4. 固定父检查点

仅使用EXP-018的U2第300轮固定检查点。

路径：

experiments/EXP-018/runs/EXP-018-CONTINUE-001/U2/epoch-300.pth

SHA256：

19b1312cbaae49ad95a7d001270c4357d7ea23b2584d19698b0084b3da4b094f

运行前及运行后验证父检查点哈希。

父检查点保持只读。

不得使用best_validation.pth替代固定第300轮模型。

不恢复训练优化器进行新训练。

本轮所有干预均从同一父模型独立开始。

## 5. 原始代码与目标模块

实际官方源码：

vendor/UniSparse/utils/utils.py

vendor/UniSparse/models/submodules/layers.py

vendor/UniSparse/models/submodules/blocks.py

vendor/UniSparse/models/cifarnet.py

原calc_penalty对SparseNeuron执行local_forward。

原活动惩罚：

rate=spike.mean(dim=(0,1,3,4))

penalty=rate.sum()*next_weight

SparseBN通过局部反向传播获得penalty_gamma和penalty_beta。

原手动更新：

gamma -= lmbda * penalty_gamma

beta -= lmbda * penalty_beta

本轮固定两个目标路径：

路径L1：

static_conv.norm -> static_conv.activation -> conv1.conv

路径L2：

conv1.norm -> conv1.activation -> conv2

先通过真实模型对象、hook和注册关系确认以上路径。

要求：

- 存在单个直接后继SparseConv2d。
- 源活动与消费者输入通道一一对应。
- 不存在中间MaxPooling或flatten。
- 目标BN具有可优化gamma、beta。
- groups=1。
- 其他消费者不存在，或明确记录其实际影响。

若实际计算图与上述条件不符，暂停该路径，不自行替换其他层。

本轮不干预其他模块。

## 6. 已有真实扇出数据

U2第300轮：

conv1.conv：

- nominal=2304
- mean=119.1875
- std=149.41693115234375
- CV=1.2536292073610382
- min=0
- P10=7.5
- P50=53.5
- P90=376.5
- max=721

conv2：

- nominal=2304
- mean=217.07421875
- CV=0.671192525665365
- min=0
- max=771

以上为忽略空间边界的实际有效连接计数。

使用实际父模型重新核查这些统计，不直接将历史数字作为本轮测量值。

## 7. 理论公式R74—R78

对目标神经元模块：

s.shape=[T,B,C,H,W]

rate[c]=mean(s,dim=(0,1,3,4))

M.shape=[C_out,C_in,k_h,k_w]

M=consumer.sparse_mask().detach()

d[c]=sum(M[:,c,:,:])

d0=C_out*k_h*k_w

d_mean=mean(d)

phi[c]=(gamma[c],beta[c])

原始活动代理：

P0=d0*sum_c(rate[c])

R74：

g0=dP0/dphi

该梯度通过原ATan替代梯度计算，不是硬阈值函数的严格导数。

R75：

a[c]=(1-rho)+rho*d[c]/d_mean

固定rho=0.25。

R76：

P_MP=d0*sum_c(a[c]*rate[c])

R77：

mean(a)=1

当d_mean=0时，设置a为全1并记录异常。

a和M均detach。

不允许梯度传播至权重掩码、权重阈值或原始卷积权重。

不得对实际d_mean随意加epsilon，破坏mean(a)=1。

使用float32或float64进行扇出统计和系数计算。

本轮不是动态mask训练，所有mask在局部干预期间固定。

R78：

g_rho=(1-rho)*g0+rho*(d0/d_mean)*h

其中h为局部sum(d[c]*rate[c])对phi的替代梯度。

该关系在相同局部输入、相同神经元状态及通道独立权重条件下检查。

仅表示局部活动代理梯度的线性组合。

不得把R78解释为真实全网络SOPs的严格梯度公式。

## 8. 实验分组

B0：

不执行任何干预，原始U2第300轮。

G0：

a[c]=1。

G1：

a[c]=0.75+0.25*d[c]/d_mean。

G2：

对G1的a向量进行固定随机置换。

每层独立使用CPU随机数生成器：

seed=20261128+目标层固定顺序索引。

保存实际permutation。

G2必须保持与G1完全相同的系数多重集合、均值、标准差和最小最大值，只改变系数与真实通道之间的对应关系。

不得根据任务结果挑选随机置换。

## 9. 数据划分

复用已锁定的数据划分：

experiments/EXP-004/splits/indices.json

SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

原训练集45000张。

原验证集4500张。

原500张旧诊断池不使用。

从45000张训练来源样本中，使用固定seed=20261128，无放回选取两个互不重叠的样本子集：

- D_grad：512张，用于估计局部活动惩罚梯度。
- D_cal：512张，用于选择参数干预强度及匹配SOP预算。

保存明确样本索引和哈希。

两者与4500验证集均无交集。

注意，这些样本此前已参与Baseline训练，因此不能称为独立未见训练样本。

所有诊断使用确定性验证预处理，不使用随机数据增强。

最终4500张验证集只用于已锁定方案的性能评价。

不根据最终验证结果重新选择干预尺度、rho或通道系数。

禁止使用官方test_batch。

## 10. 本轮环境与运行配置

沿用ROUND-017通过审计的UniSparse专用环境。

Python 3.11.9。

PyTorch 2.4.1+cu118。

CuPy 13.5.1。

SpikingJelly 0.0.0.0.14及已记录的独立兼容补丁。

模型：

CifarNet。

base_channels=256。

T=4。

原CuPy LIF。

建议诊断Batch Size=64。

num_workers=4。

seed=20261128。

无优化器训练。

全部正式比较采用相同FP32推理及局部梯度精度。

首先重新评价B0，并与历史AMP训练期间的验证结果比较。

如存在精度口径差异，如实记录；本轮主要比较以重新测量的同口径B0为准。

不得混用FP32与AMP结果作为公平对照。

## 11. 局部活动梯度计算协议

本轮目的是隔离活动惩罚方向，因此使用冻结BN运行统计的eval局部重放。

该机制不同于作者正式训练中使用batch统计的SparseBN.local_forward。

不得宣称本诊断完全复现原训练中的calc_penalty。

步骤：

1. 加载父检查点。
2. 设置model.eval()。
3. 冻结全部卷积权重、阈值及其他非目标参数。
4. 在D_grad上以no_grad执行真实完整前向。
5. 通过hook获取目标BN的真实归一化前输入。
6. 对每个Batch执行必要的functional.reset_net。
7. 逐批局部重放目标BN和原CuPy LIF。
8. 使用目标BN保存的running_mean、running_var和eps。
9. gamma、beta为唯一允许局部梯度的参数。
10. F.batch_norm使用training=False。
11. 使用原神经元tau、阈值、重置方式及ATan替代梯度。
12. 分别计算G0、G1、G2的活动惩罚及局部梯度。
13. 及时释放计算图及中间状态。
14. 按各Batch实际样本数加权累积局部梯度。
15. 不在梯度估计期间实际改变模型参数。

必须检查局部重放的脉冲序列与同一BN输入下的真实网络对应输出一致。

如果不一致，先核查膜电位初始化、BN统计、精度及计算图。

不得直接使用未经验证的重放结果。

避免保存512张完整多时间步特征张量。逐批采集、逐批计算、在线汇总。

不使用全网络任务反向传播，不更新卷积权重。

## 12. R79：参数更新范数匹配

对每个目标层l，将gamma和beta的局部梯度连接成向量：

g[m,l]=concat(grad_gamma,grad_beta)

计算：

g_tilde[m,l]=g[m,l]*norm(g[0,l])/norm(g[m,l])

G0保持原方向。

对于norm(g0)=0的层，三个干预组均设置该层更新为0并记录。

若norm(g0)>0但某组g范数为0或非有限，则标记该组机制检查失败，不通过任意epsilon修补。

检查每层实际更新方向的L2范数匹配。

建议FP32归一化误差不超过1e-5。

记录：

- 原始梯度范数。
- 范数匹配后的梯度范数。
- 原方向和候选方向余弦相似度。
- 有效扇出与原始/候选梯度幅度的Spearman相关。
- R78梯度线性组合误差。

可以额外计算使用实际空间有效扇出的局部成本梯度用于只读诊断，但不得改变本轮干预方向。

所有梯度统计均标记为局部eval代理，不作为正式训练动力学结论。

## 13. R80：受控参数干预

将两个目标层的gamma、beta连接形成总向量phi。

连接两个目标层的G0梯度形成g0。

定义相对参数干预尺度q：

eta(q)=q*norm(phi)/norm(g0)

各组：

phi_new[m,l]=phi_parent[l]-eta(q)*g_tilde[m,l]

这样在相同q下，各组目标参数的整体更新范数相同。

固定尺度集合：

q in {0,0.0025,0.005,0.01,0.02,0.04,0.08,0.16}

q为本轮探索性诊断尺度，不是论文正式训练学习率。

每次试验均从同一原始父模型恢复。

不得累积应用多次干预。

干预只能改变两个指定BN的gamma和beta。

每次干预后核查：

- 全部卷积原始权重不变。
- 全部稀疏阈值不变。
- 实际有效权重掩码不变。
- BN running_mean/running_var不变。
- 非目标gamma/beta不变。
- 参数均有限。
- 神经元状态正确reset。

保留各次实际参数差分审计。

## 14. EXP-020A：同尺度机制验证

首先在D_cal上重新测量B0。

然后在固定q集合上分别评价G0、G1、G2。

记录：

- 实际BN affine更新范数。
- 目标两个消费者卷积的SOPs。
- 全网络SOPs。
- 目标两个神经元模块活动率。
- 各目标模块的通道活动变化。
- CE。
- Accuracy。

H-019的主要机制检查点固定为q=0.02。

在该尺度上比较G1、G0及G2的目标两个消费者合计SOP下降量。

若G1的真实目标消费者SOP下降量大于G0与G2，则方向预测获得初步支持。

若相反，则记录反对证据。

其余尺度只作为预设响应曲线观察，不挑选有利尺度替代主要检查点。

实际SOP必须通过真实完整模型前向测量。

不能直接将加权局部活动代理下降视为真实计算收益。

## 15. EXP-020B：相同实际SOP预算筛选

定义相对同口径B0的实际SOPs降幅：

R_m(q)=1-C_m(q)/C_B0

在D_cal上，预设目标为5%降幅。

每组从固定q集合中选择：

q_m*=argmin_q abs(R_m(q)-0.05)

只允许选择降幅位于[0.04,0.06]的候选。

若距离相同，选择较小q。

不使用准确率或CE作为q选择依据。

若某组不存在满足条件的q：

- 标记该组无法预算匹配。
- 保留全部尺度响应。
- 不扩大q搜索范围。
- 不进行插值后额外实验。
- 不使用验证集选择新的q。

锁定各组实际q、系数、置换、梯度版本及D_cal结果后，再运行4500张正式验证样本。

仅评价B0和各组已锁定候选。

## 16. 最终性能指标

使用固定4500张验证集。

每组报告：

- 验证准确率。
- 平均CE。
- 全局加权FR。
- 逐层真实消费者输入活动率。
- SOP_author。
- SOP_sample_weighted。
- 理论SOPs相对B0降幅。
- 有效权重数量及稀疏率。
- 参数更新范数。
- 各层SOPs。
- 逐样本真实预测及CE。

SOP_sample_weighted：

sum(所有真实Batch的SOP总数)/实际样本总数。

验证尾批保留。

采用与EXP-019一致的7层监控范围。

验证实际监控输入均为二值脉冲。

不混淆理论SOPs、实际硬件能耗及真实GPU加速。

## 17. 预先锁定的推进门槛

G1进入下一轮正式短程训练候选，需要：

1. 必要数学、状态和计算检查全部通过。
2. H-019主要检查点方向预测通过。
3. G1验证集SOP降幅在4%至6%。
4. G0/G2均取得匹配候选，且与G1的SOP降幅差异分别不超过1个百分点。
5. G1相对B0准确率下降不超过0.5个百分点。
6. G1相对G0准确率提高至少0.3个百分点。
7. G1相对G2准确率提高至少0.3个百分点。

以上为探索性资源推进标准，不代表统计显著性。

报告G1对G0、G2的逐样本配对Bootstrap 95%置信区间。

固定重采样次数2000，seed=20261128。

准确率差使用逐样本正确性差异。

CE差使用逐样本CE差异。

保留区间跨零和不利结果，不调整统计方法。

如果全部门槛通过，只能推荐下一轮正式训练干预，不自动启动训练。

若机制方向正确但性能门槛失败，记录局部计算效率与任务性能之间的冲突。

若G1未优于G2，不宣称有效连接信息具有特异性收益。

若预算不匹配，标记性能假设证据不足。

## 18. 最低正确性测试

至少完成：

- rho=0严格恢复原始惩罚方向。
- 均匀有效扇出条件下a=1。
- 非均匀有效扇出条件下mean(a)=1。
- d_mean=0正确回退。
- 所有a有限且非负。
- G1/G2系数集合一致。
- R78梯度线性组合成立。
- R79实际范数匹配成立。
- q=0完全恢复父模型输出。
- 捕获开关不改变完整模型输出。
- 局部重放神经元状态正确reset。
- 父检查点哈希不变。
- 全部有效权重掩码不变。
- BN运行统计及非目标参数不变。
- SOP监控与独立参考计数一致。
- 数据划分及样本配对正确。

合成测试与真实模型实验结果分别保存。

如果关键正确性检查失败，不进入正式性能评价。

## 19. 代码修改范围

保持vendor/UniSparse原始代码不变。

优先新增独立模块：

experiments/ROUND-018/

包含：

- 局部活动梯度测量。
- 有效扇出系数计算。
- 范数匹配。
- 单次参数干预。
- 固定尺度预算匹配。
- 验证及统计。
- 单元测试。

具体文件名依据实际代码结构决定。

不进行无关重构。

不将MP-EFR直接写入正式Baseline训练入口。

不修改神经元动力学、替代梯度、稀疏阈值规则或优化器。

## 20. 资源预算及停止条件

需要用户新授权。

建议：

- 1张符合实验室共享管理规定的RTX4090。
- GPU总占用上限30分钟。
- CPU累计计算上限60分钟。
- 正式训练Epoch=0。
- 不执行任何额外GPU实验。

所有前置检查、真实前向、局部反向及评估均计入GPU预算。

优先选择实际显存充足的单GPU。

不得干扰其他用户的进程。

出现以下情况停止相关实验：

- 父检查点哈希不一致。
- 数据隔离失败。
- 公式实现不一致。
- 实际权重掩码发生改变。
- BN运行统计发生未经授权更新。
- 非有限状态。
- OOM。
- 超出资源预算。

如需改变Batch Size或新增恢复措施，记录实际原因及对照影响；无法保持公平时停止并报告。

不得自行扩大q集合、rho集合、目标模块、数据集或训练轮数。

## 21. 科研状态更新

追加：

research/ROUND-018-TASK.md

更新：

- THEORY.md，追加v0.2.4及R74—R80。
- HYPOTHESES.md，更新H-016/H-018限定判定，新增H-019。
- RESEARCH_STATE.md。
- UNISPARSE_BASELINE.md。
- EXPERIMENT_PLAN.md。
- EXPERIMENT_REGISTRY.csv。
- DECISION_LOG.md，追加DEC-018。
- LITERATURE.md。

保留EXP-018和EXP-019为COMPLETED。

H-017保持待验证。

根据项目规范更新AGENTS.md中当前Baseline身份的过时描述：AT-LIF是已归档Baseline，UniSparse为当前Controlled Baseline。保留原科研规范及历史版本。

不修改最高科研诚信、预算及状态管理原则。

## 22. 文献及创新性限制

保存以下相关文献及真实链接：

https://doi.org/10.3389/fnins.2020.00662

https://doi.org/10.3389/fnins.2022.760298

https://ieeexplore.ieee.org/document/11132570/

https://proceedings.mlr.press/v306/shi26a.html

这些研究已经涉及SynOps正则化、权重及活动联合稀疏、名义扇出或动态联合优化。

不得宣称简单的归一化、凸组合或者有效扇出加权是学术首创。

当前MP-EFR只是待验证的具体增量方法。

## 23. 实验交付

建立独立experiments/EXP-020目录及唯一Run ID。

至少保存：

- TASK.md
- config.json
- environment.txt
- code_version.json
- commands.sh
- status.json
- parent_checkpoint_audit.json
- data_split.json
- target_dependency_audit.json
- fanout_vectors.csv
- coefficient_vectors.csv
- permutation.json
- local_replay_tests.json
- gradient_formula_tests.json
- gradient_statistics.csv
- update_norm_audit.csv
- parameter_integrity_audit.json
- calibration_response.csv
- selection_lock.json
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

图表：

1. 两个目标路径的真实有效扇出分布。
2. G0/G1/G2惩罚系数分布。
3. 不同q下目标消费者SOP下降曲线。
4. 不同q下全网络SOP下降曲线。
5. 干预尺度与CE/Accuracy关系。
6. 锁定候选的准确率—SOP权衡。

若某阶段未执行，明确标记NA及实际原因。

不生成虚构数据。

保留所有失败和重试记录。

## 24. GitHub最终交付

从上轮最终结果SHA建立独立研究分支。

读取并遵守AGENTS.md。

正式实验前提交理论、任务、固定配置、数据划分、判定规则及执行代码。

实验结束后提交全部真实结果。

不修改main，不force push，不删除历史实验。

最终返回：

- GitHub实验分支链接。
- 实际执行代码SHA。
- 最终结果提交SHA。
- EXP-020实际执行状态。
- 数学及状态检查结果。
- H-019主要q=0.02机制结果。
- 各组完整尺度响应。
- 各组实际锁定q。
- G0/G1/G2真实验证准确率、CE、FR、SOPs及稀疏率。
- 同预算匹配检查结果。
- 配对Bootstrap置信区间。
- 预注册推进门槛结果。
- 全部异常、限制及负面数据。
- 实际GPU和CPU资源消耗。
- REPORT及原始数据路径。
- 是否值得进入下一轮正式训练干预的工程建议。

不得根据合成梯度检查、局部代理下降或者单个有利尺度宣称新方法已经有效。

本轮目标是验证一个明确、可证伪的创新机制，而不是直接追求新的最高准确率。

获得用户资源授权后执行。