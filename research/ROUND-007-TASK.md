# Codex 科研执行任务：ROUND-007

# 前部 AT-LIF 动力学扰动的独立样本验证与条件性反事实诊断

## 一、任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-007
- 科研阶段：阶段1—2
- 科研决策：DEC-007，REFINE + REPLICATE
- 理论版本：v0.1.3
- 原主线：Q-001
- 新细化问题：Q-001A，前部模块动力学输出偏差的任务作用
- 原科研假设：H-001、H-002
- 新科研假设：H-004、H-005
- 本轮实验：EXP-005、条件性EXP-006
- GitHub仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-006-dynamics-diagnostics
- 本轮建议分支：research/round-007-independent-mechanism
- 上轮执行代码SHA：a7e0e9cc0efe1ea69122aaf4270e7f462ec93fad
- 上轮结果SHA：ee0624d8af345feaa39e44ca0c6508b0d30bca41
- 依赖实验：EXP-001、EXP-002、EXP-004
- 优先级：P0独立样本验证；P1条件性反事实诊断

首先读取AGENTS.md、THEORY.md、HYPOTHESES.md、RESEARCH_STATE.md、ROUND-001-TASK.md、ROUND-006-TASK.md和EXP-002报告及真实原始数据。

本轮不训练、不微调、不修改Baseline核心算法、不使用官方测试集。

## 二、ROUND-006正式科研结论

理论研究员已经读取GitHub原始数据、实际代码及实验报告，完成独立统计。

EXP-002有效完成。

原主要条件：

conv3_x.1.residual_function.2，alpha=1.5，256个固定诊断样本。

真实结果：

- R_FR=0.5977283898930732。
- mean(E)=0.026041029432550853。
- E的95% CI=[0.02574658759203404,0.026323316503716793]。
- 平均交叉熵变化=-0.024405735373107085。
- 95% CI=[-0.09605079152643584,0.0389564814817811]。
- 新增静默比例=0.5570208977842445。
- R1/R2最大归一化误差低于1e-5。
- 固定有效输入及诊断控制通过。

根据原预注册条件，H-001正式判定为：

CURRENT_EVIDENCE_AGAINST，在原主要模块、原模型、原指标和原干预条件下反对具有预设实际幅度的假设。

不得扩大为所有层及所有AT-LIF网络均不存在动力学偏差。

H-002保持UNTESTED。

原EXP-003保持BLOCKED，未达到进入门槛。

不得事后更换原EXP-002的主要模块、指标或判断标准。

## 三、ROUND-006补充探索结果

alpha=1.5：

前部模块conv1.2：

- R_FR=0.4429347649668636。
- mean(E)=0.16814114183212184。
- E的95% CI=[0.16660827392475122,0.16967269200339105]。
- 平均损失变化=0.1416489813888328。
- 配对95% CI=[0.010684706578814781,0.2855079504815722]。
- 准确率由0.8984375降至0.8515625。
- 新增静默比例=0.3068096759159379。

后部模块conv4_x.2.act：

- mean(E)=0.1825099736442321。
- 平均损失变化=-0.014200462737048641。

前部结果提供进一步验证的理由，但仍属于已经观察过的探索性结果。

不能用原256个样本重复实验后宣称获得独立样本证据。

## 四、理论研究员新增数学分析

保留R1—R14及原预注册协议。

新增R15—R20，理论版本v0.1.3。

### R15—R17：归一化尺度分析

定义：

N[b] = sum_i(abs(O_alpha[b,i]-O_ref[b,i]))

I[b] = sum_t,i(abs(x_ref[t,b,i]))

A[b] = sum_i(abs(O_ref[b,i]))

原指标：

E[b] = N[b]/I[b]

新增参考输出归一化指标：

R15: E_out[b] = N[b]/A[b]

R16: q[b] = A[b]/I[b]

当两个分母均非零时：

R17: E[b] = q[b]*E_out[b]

这是逐样本代数恒等式，不要求跨样本均值满足均值乘积关系。

上轮真实数据重新计算：

前部mean(E_out)=0.41272438718878507，mean(q)=0.4073116155969185。

中部mean(E_out)=0.6377558808014446，mean(q)=0.04103719017906809。

后部mean(E_out)=0.6137705574556233，mean(q)=0.29707795988795715。

说明原E存在跨模块归一化尺度差异。

新增指标仅用于辅助解释，不改变原R5主检验标准，不将H-001重新判定为成立。

### R18：累计输出总量关系

对于正的模块共享标量阈值：

O_ref=theta*K_ref

O_alpha=alpha*theta*K_alpha

在统计范围一致时：

R18:

output_retention = sum(O_alpha)/sum(O_ref)

= alpha*(1-R_FR)

该关系衡量总量，不代表逐位置累计输出完全一致。

验证真实脉冲计数和输出统计满足该式。

### R19：新增静默导致的误差恢复上界

设N_silent[b]为K_ref>0且K_alpha=0的位置数量。

因为每个新增静默位置原本至少发放一次：

E_floor[b] >= theta*N_silent[b]/I[b]

所以：

R19:

R_E <= 1-mean(theta*N_silent/I)/mean(E_G1)

在上轮数据中的宽松理论上界：

- 前部约0.67023。
- 中部约0.29315。
- 后部约0.33285。

这些不是实际补偿结果。

它们仅是在原有限幅值补偿规则下的可恢复性上界。

### R20：最后一层精确读出关系

实际ResNet20最后一个ATLIF之后为全局平均池化及线性分类器，最终任务使用时间平均logits。

因此：

Delta_z = W*GAP(Delta_O)/T

其中Delta_O是最后一层沿时间求和后的输出差。

此关系仅适用于当前最后一层及线性读出条件，不推广至存在下游非线性动力学的前部层。

本轮不额外开展后部实验，只记录该竞争性解释。

## 五、新科研假设

H-004，初始状态UNTESTED：

当前训练完成的ResNet20 AT-LIF模型，在前部模块conv1.2的阈值提高至1.5倍时，会产生具有实际幅度的累计输出偏差及任务损失增加。

H-005，初始状态UNTESTED：

若H-004获得独立样本支持，在保持受干预模块脉冲及膜电位轨迹不变时，参考输出引导的逐位置幅值补偿能够恢复部分累计输出并缓解部分任务损失，且该作用不只是一般性的幅值缩放。

两项假设均只针对当前模型及固定干预条件。

本轮不能验证独立训练种子间的稳定性，不能证明所有AT-LIF层具有相同问题，也不能证明原模型训练阈值本身存在缺陷。

## 六、固定模型及独立验证样本

使用第194轮最佳检查点：

experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth

SHA256：

c068f54379cb5d29bcfa5fe2d9de676cd3cc153433ada1ed5aa1c66355d0b32a

原数据划分：

experiments/EXP-004/splits/indices.json

SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

读取：

diagnostic：500个官方训练集原始索引。

diagnostic_256：ROUND-006使用的256个索引。

本轮候选验证样本：

remaining_ids = sorted(set(diagnostic)-set(diagnostic_256))

预期244个。

核查此前全部实验的数据访问记录，确认剩余样本是否被其他任务使用。

如果发现已访问样本，记录并排除。

不得根据模型正确性、损失、类别或干预结果挑选样本。

如果没有足够证据证明剩余样本未被此前机制诊断使用，必须明确说明独立性限制。

保存本轮实际索引及SHA256，并在读取正式诊断结果前提交预注册配置。

不能重新生成原数据划分。

剩余诊断数据使用后，应标记诊断池已被用于机制分析，不能把它重新作为后续算法的独立最终评价集。

## 七、EXP-005：独立样本验证

主要条件：

- module=conv1.2。
- alpha=1.0和1.5。
- T=8。
- Batch Size=32。
- FP32。
- model.eval()。
- torch.no_grad()。
- 完全冻结权重。
- 使用固定评估预处理。

不增加其他alpha或干预模块。

复用ROUND-006已经通过验收的诊断实现。

注意现有diagnostics.py的主入口使用固定256个样本及旧输出目录，需要新增独立入口或最小参数化接口。

禁止覆盖EXP-002原始数据。

保留旧脚本及旧执行结果。

### 必要正确性检查

检查点与数据哈希正确。

alpha=1.0时诊断开启与关闭模型输出一致。

目标模块输入在alpha=1.0和1.5时完全相同。

脉冲由原始输出及阈值判断获得。

R1/R2归一化残差容差1e-5。

所有组模型状态、阈值、BN及ATLIF内部统计恢复正确。

不重复完整EXP-001合成测试。

### 指标

沿用原R4—R8公式：

R_FR、E、脉冲翻转率、新增静默比例、逐样本交叉熵差异。

新增辅助指标：

E_out、q、R18总量保留比例、R19宽松恢复上界。

保存全部逐样本原始分子、分母、计数及损失。

采用2000次样本级配对Bootstrap，seed=2027。

记录无效分母、无效样本及无效重采样次数。

### H-004预设支持条件

所有以下条件同时满足：

1. R_FR>=0.20。
2. mean(E)>=0.05。
3. mean(E)的95% CI下界>0.05。
4. mean(loss_G1-loss_G0)>=0.02 nats。
5. 任务损失配对95% CI下界>0。
6. 所有诊断完整性及数值检查通过。

若条件未满足，不得更换层或alpha继续寻找阳性结果。

如果CI明确排除预设效应，记录反对证据。

若CI过宽，记录证据不足。

只报告实际达到或未达到的条件，由理论研究员正式更新假设状态。

## 八、反事实可恢复性预检查

在EXP-005主要条件中，利用现有累计输出张量计算：

eligible[b,i] = (K_ref[b,i]>0) AND (K_alpha[b,i]>0)

c_star[b,i] = clip(O_ref[b,i]/O_alpha[b,i],0.25,4)，适用于eligible位置。

其他位置c_star=1。

定义最小可达到的累计输出误差：

E_clip_min[b] = sum_i(abs(c_star[b,i]*O_alpha[b,i]-O_ref[b,i]))/I[b]

对于正阈值及规定系数区间，c_star逐位置最小化上述绝对累计输出误差。

因此：

R_E_attainable = 1-mean(E_clip_min)/mean(E_G1)

这属于原补偿规则下考虑裁剪的实际可达误差恢复比例，不需要执行G2完整下游前向即可计算。

同时计算R14和R19，并检查：

R_E_attainable <= R14上界 <= R19宽松上界

允许合理数值舍入误差。

如果H-004门槛未通过，或R_E_attainable<0.50，则不执行EXP-006。

不得通过放宽clip范围、改变eligible定义或增加新脉冲规避门槛。

## 九、EXP-006：条件性反事实诊断

仅当EXP-005全部进入条件通过，并且原补偿规则具备至少50%误差恢复能力时执行。

分组：

G0：原阈值，原输出。

G1：1.5倍阈值，原输出。

G2：1.5倍阈值，逐位置参考累计输出补偿。

G3：1.5倍阈值，随机置换G2的eligible位置系数。

G4：1.5倍阈值，样本级统一幅值补偿。

G2和G3严格沿用原ROUND-001/006反事实公式及置换规则。

G3随机种子2028。

新增G4：

对于每个样本：

c_global[b] = clip(
sum_{eligible i}(O_ref[b,i]) /
sum_{eligible i}(O_alpha[b,i]),
0.25,4)

所有eligible位置使用相同c_global[b]。

非eligible位置保持系数1。

若不存在eligible位置，则所有系数为1并记录原因。

G4只作为一般性幅值调整的竞争解释对照。

所有组必须使用相同模型、输入、数据顺序、初始状态、设备和推理精度。

G1—G4受干预模块自身完整二值脉冲张量必须逐元素一致。

实际有效输入必须一致。

重置前膜电位及重建重置后膜电位采用既定数值容差检查。

所有补偿必须发生在目标ATLIF前向输出之后的下游传递边界。

不改变目标模块自身的动力学。

不将反事实补偿后的输出重新解释为二值脉冲。

G0/G1可以复用EXP-005已验证一致的结果。

### 指标

沿用R11、R12：

R_E=(mean(E_G1)-mean(E_G2))/mean(E_G1)

R_loss=(mean(loss_G1)-mean(loss_G2))/
(mean(loss_G1)-mean(loss_G0))

同时报告：

Gain_21=mean(loss_G1-loss_G2)

Gain_23=mean(loss_G3-loss_G2)

Gain_24=mean(loss_G4-loss_G2)

使用2000次样本级配对Bootstrap，seed=2027。

各组保存逐样本损失、准确率、累计输出误差、补偿覆盖率、静默比例、系数分布和裁剪比例。

检查实际R_E与前置计算的R_E_attainable一致。

### H-005初步支持条件

- R_E>=0.50。
- R_loss>=0.25。
- Gain_21的95% CI下界>0。
- Gain_23的95% CI下界>0。
- 目标模块动力学控制通过。
- 数据及指标计算有效。

此外，必须单独报告Gain_24。

只有当Gain_24的95% CI下界>0时，才能进一步认为逐位置参考补偿相对于统一幅值补偿具有初步的特异性任务收益。

若G2优于G1/G3但未明确优于G4，则不能认定复杂的逐位置补偿具有独立必要性。

若累计输出充分恢复，但R_loss有效95% CI上界仍低于0.25，记录反对具有预设任务恢复幅度的证据。

若随机补偿未形成有效置换、分母不稳定或其他控制失败，记录证据不足，不作强结论。

## 十、资源与终止条件

本轮只允许冻结模型诊断。

建议用户批准：

- 一张符合实验室共享使用规则的RTX 4090。
- GPU总占用上限15分钟。
- CPU计算累计不超过30分钟。

GPU未授权时不得启动GPU计算。

不执行任何形式的训练、微调、超参数搜索或新方法实现。

实验顺序：

独立样本核查及预注册 > 必要工程检查 > EXP-005 > 可恢复性门槛 > 条件性EXP-006。

如果EXP-005未达到门槛，直接结束本轮主要实验。

不增加其他模块、alpha、归一化门槛或备用样本寻找支持结果。

## 十一、原始数据与交付

为EXP-005及EXP-006建立独立实验目录和Run ID。

保存：

- TASK.md。
- config.yaml。
- environment.txt。
- code_version.json。
- commands.sh。
- status.json。
- 实际使用的诊断索引及哈希。
- per_sample.csv。
- metrics.json。
- summary.csv。
- run.log。
- figures/。
- REPORT.md。
- limitations.md。

原始逐样本数据应足以让理论研究员独立复算全部主要指标、置信区间及可恢复性上界。

未运行实验的结果保持NA。

报告区分：

1. 原ROUND-006探索性证据。
2. ROUND-007独立样本验证证据。
3. 尚未验证的竞争性解释。

如EXP-006被阻塞，记录具体未达到的门槛。

## 十二、科研状态更新

新增research/ROUND-007-TASK.md。

THEORY.md追加v0.1.3及R15—R20，不覆盖旧理论。

HYPOTHESES.md：

- H-001：CURRENT_EVIDENCE_AGAINST，限定原主要条件。
- H-002：UNTESTED。
- H-003：INSUFFICIENT_EVIDENCE。
- H-004：UNTESTED，待本轮验证。
- H-005：UNTESTED，条件性验证。

保留：

- EXP-002 COMPLETED。
- EXP-003 BLOCKED及其原始原因。
- EXP-004 COMPLETED。

新增EXP-005、EXP-006。

追加DEC-007：REFINE + REPLICATE。

记录原主线不能作为普遍性累计输出补偿不足问题继续推进。

本轮是针对前部探索结果的有限独立验证，不允许无限更换干预条件。

LITERATURE.md补充以下文献：

1. Wang et al., Adaptive Calibration: A Unified Conversion Framework of Spiking Neural Network, https://arxiv.org/abs/2412.16219

2. Shi, Bu, Yu, UniSparse: Combining Weight Pruning and Spike Sparsification in Spiking Neural Networks, ICML 2026, https://proceedings.mlr.press/v306/shi26a.html

3. Optimizing the Energy Consumption of Spiking Neural Networks for Neuromorphic Applications, 2020, https://doi.org/10.3389/fnins.2020.00662

标记为初步创新性风险排查，不宣称完成系统查重。

## 十三、GitHub版本及执行

开始前核查实际仓库和AGENTS.md。

从上轮结果SHA创建独立分支。

正式读取独立样本干预结果前，提交本轮任务、配置、索引和执行代码。

记录真实执行SHA。

通过必要检查且已有用户资源授权后直接执行。

保留历史结果，不修改main，不force push。

实验完成后提交全部真实指标、日志、报告和科研状态。

## 十四、最终返回信息

- GitHub分支链接。
- 实际执行代码SHA。
- 结果提交SHA。
- 独立验证实际样本数量。
- 数据独立性检查。
- EXP-005执行状态。
- R_FR及CI。
- mean(E)及CI。
- mean(E_out)及mean(q)。
- 任务损失变化及CI。
- R14、R19及考虑clip的实际可达恢复比例。
- H-004预设门槛是否满足。
- EXP-006是否执行。
- G0—G4各组实际结果。
- R_E及R_loss。
- Gain_21、Gain_23、Gain_24及CI。
- 动力学控制检查。
- 实际资源消耗。
- 原始数据及报告路径。
- 全部异常与局限。

本轮不是为了挽救原H-001，而是严格判断前部模块的探索性发现是否值得发展为新的、范围明确的科研问题。

实验可以支持、反对或无法判断新假设。禁止编造任何数据，禁止根据预期结论修改实验。