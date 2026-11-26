# THEORY v0.1.0

以下为本轮预先提交的完整定义；原文按节保存，非已有实验证据。

## 四、独立理论公式

以下公式属于本项目 THEORY v0.1.0 的独立代数推导，不是原论文的新定理。

定义：

```text
X = sum_t(x[t])

O = sum_t(theta * s[t])

D_leak = (1 - tau) * sum_t(u[t-1])

D_reset = sum_t(s[t] * (m[t] - theta))

D_res = u[T] - u[0]
```

根据原始 AT-LIF 动力学：

```text
R1:
X = O + D_leak + D_reset + D_res
```

对于相同有效输入序列、相同初始状态、相同泄漏系数，仅改变发放阈值的两条神经元轨迹：

```text
theta_alpha = alpha * theta_0

Delta_O = O_alpha - O_0
Delta_D_leak = D_leak_alpha - D_leak_0
Delta_D_reset = D_reset_alpha - D_reset_0
Delta_D_res = D_res_alpha - D_res_0

R2:
Delta_O = -Delta_D_leak
          -Delta_D_reset
          -Delta_D_res
```

成立前提：

- 离散膜电位动力学与上述公式一致。
- 硬重置至零。
- theta > 0。
- 单个样本仿真期间阈值固定。
- 输出为 theta * s。
- 正确捕获神经元实际有效输入和初始状态。

注意：

- 允许负输入及负膜电位。
- D_leak 和 D_res 可能为负。
- D_reset 在正阈值且正常发放条件下非负。
- 各动力学项可能相互抵消。
- 不得把这些变量称为物理能量。
- 不得把累计输出偏差直接称为任务信息损失。
- 如果实际代码采用不同充电、重置、截断或初始状态规则，必须记录差异并暂停不适用的机制统计。
- 不允许为了满足公式而偷偷修改原始 Baseline。

## 五、科研假设

### H-001：动力学补偿偏差假设

状态：UNTESTED。

在冻结训练权重、固定有效输入的条件下，提高 AT-LIF 发放阈值并有效降低发放率时，可能产生具有实际幅度的累计输出偏差。

预期可观测现象：

- 发放率下降。
- 累计输出发生变化。
- 动力学分解项发生变化。
- R2 能够解释累计输出变化。

支持或反对该假设的条件见后续实验门槛。

### H-002：累计输出偏差的任务影响假设

状态：UNTESTED。

当提高阈值导致任务损失增加时，累计输出偏差可能是部分性能下降的可干预原因。

如果在不改变受干预模块脉冲状态和发放时序的条件下，恢复部分参考累计输出能够缓解任务损失，则为该假设提供初步支持。

竞争性解释：

- 脉冲时序变化。
- 神经元完全静默。
- 一般幅值调节。
- 下游非线性响应。
- 网络层敏感性。
- 训练后权重适应。

不能将预期结果作为必须产生的结果。


### 7.3 数值误差

```text
R3:

residual = X - O - D_leak - D_reset - D_res

e_id = abs(residual) / (
    1 + abs(X) + abs(O)
      + abs(D_leak)
      + abs(D_reset)
      + abs(D_res)
)
```

报告逐神经元最大值、均值及失败比例。

暂定容差：

- 独立 float64 模拟：1e-10。
- 实际 float32 实现：1e-5。

不得只在全部神经元求和后检查恒等式，否则错误可能相互抵消。

如果精度或算子导致容差不适用，报告并请求理论确认，不得静默修改标准。


### 8.6 评价指标

指标一：平均发放率。

```text
FR_alpha = mean(s_alpha)
```

均值覆盖样本、时间及该模块所有神经元状态。

```text
R4:
R_FR = (FR_ref - FR_alpha) / FR_ref
```

参考 FR 为零时记录 NA。

指标二：累计输出偏差。

```text
O_ref[b,i] = sum_t(o_ref[t,b,i])
O_alpha[b,i] = sum_t(o_alpha[t,b,i])

R5:
E[b] =
    sum_i(abs(O_alpha[b,i] - O_ref[b,i]))
    /
    sum_i,t(abs(x_ref[t,b,i]))
```

先沿时间维度计算每个神经元的累计输出，再计算绝对差异，不能先在神经元维度求和。

分母为零时标记 NA 并统计出现次数。

另外报告以参考累计输出绝对总量为分母的辅助指标，分母为零时同样标记 NA。

指标三：动力学分解。

逐样本、逐神经元计算：

```text
D_leak
D_reset
D_res
Delta_D_leak
Delta_D_reset
Delta_D_res
Delta_O
```

报告有符号均值、绝对值均值、中位数、P90、P95。

保留 R1、R2 恒等式检查结果。

指标四：脉冲状态翻转率。

```text
R6:
F_flip = mean(s_alpha != s_ref)
```

分开报告 1->0 和 0->1 的事件数及比例。

指标五：新增静默比例。

```text
K_ref[b,i] = sum_t(s_ref[t,b,i])
K_alpha[b,i] = sum_t(s_alpha[t,b,i])

R7:
F_silent =
    count((K_ref > 0) AND (K_alpha == 0))
    /
    count(K_ref > 0)
```

分母为零时标记 NA。

指标六：任务性能变化。

```text
R8:
Delta_loss[b] = loss_alpha[b] - loss_ref[b]

Delta_accuracy = accuracy_alpha - accuracy_ref
```

使用逐样本交叉熵和分类正确性，保留配对索引。

如额外计算 SOPs，必须记录计数定义，区分活动事件与有效非零连接，不得将其描述为硬件实测能耗。

### 8.7 统计方法

- 固定相同诊断样本。
- 保留逐样本配对记录。
- 对主要指标使用2000次样本级配对 Bootstrap。
- Bootstrap 随机种子：2027。
- 报告95%百分位置信区间。
- 对 R_FR 等比例指标，在每次重采样后重新计算整体分子和分母。
- 不将神经元数量当作独立样本数量。
- 不将样本级置信区间解释为独立模型训练种子之间的稳定性。
- 明确报告 NA 数量及排除范围。


### 9.3 补偿公式

对每个样本 b、神经元 i：

```text
O_ref[b,i] = sum_t(o_ref[t,b,i])
O_alpha[b,i] = sum_t(o_alpha[t,b,i])

K_ref[b,i] = sum_t(s_ref[t,b,i])
K_alpha[b,i] = sum_t(s_alpha[t,b,i])

eligible[b,i] =
    (K_ref[b,i] > 0) AND
    (K_alpha[b,i] > 0)
```

补偿系数：

```text
R10:

if eligible[b,i]:
    c[b,i] = clip(
        O_ref[b,i] / O_alpha[b,i],
        0.25,
        4.0
    )
else:
    c[b,i] = 1.0
```

补偿输出：

```text
R9:
o_cf[t,b,i] = c[b,i] * o_alpha[t,b,i]
```

正阈值条件下，eligible 区域的 O_alpha 理应为正。

遇到不符合该条件的数值异常应报告，不得任意添加 epsilon 掩盖错误。

c 的逻辑维度为 [B,N] 或与实际空间状态对应的 [B,C,H,W]。

沿时间维度广播，不沿错误的 Batch 或通道维度广播。

系数在无梯度模式下计算。

只修改传递给下游网络的输出，不改变受干预神经元自身的 s、m、u 和重置轨迹。

如果模块输出同时被用于自身重置或反馈计算，必须找到独立的下游传递边界；无法隔离时暂停实验。

### 9.4 随机补偿对照

G3 在同一样本、同一个模块内部，仅对 eligible 区域中的补偿系数做固定种子的随机置换。

随机种子：2028。

不改变系数分布，不使用标签。

不符合 eligible 条件的位置保持 c=1。

若可置换元素少于2个，记录该样本无法形成有效置换对照。

### 9.5 必须通过的对照检查

G1、G2、G3 的受干预模块：

```text
s_G1 == s_G2 == s_G3
```

要求完整脉冲张量逐元素一致。

同时验证该模块自身膜电位及重置轨迹一致。

G2、G3 不允许新建脉冲或删除原有脉冲。

下游网络的脉冲及膜电位状态允许自然变化。

报告：

- eligible 神经元比例。
- 原参考活动神经元中的补偿覆盖率。
- 系数截断比例。
- 补偿系数分布。
- 新增静默比例。
- 未能恢复累计输出的原因。

注意：R1、R2 适用于原始神经元内部输出；反事实输出是额外下游干预，不要求其继续满足原始动力学平衡式。

### 9.6 评价指标

G0、G1、G2、G3 均报告：

- 逐样本交叉熵。
- 分类准确率。
- 主要模块累计输出偏差。
- 主要模块 FR。
- 主要模块脉冲状态一致性。

定义：

```text
R11:
R_E = (mean(E_G1) - mean(E_G2)) / mean(E_G1)
```

以及：

```text
R12:
R_loss =
    (mean(loss_G1) - mean(loss_G2))
    /
    (mean(loss_G1) - mean(loss_G0))
```

任务损失增量非正或过小时，不使用 R_loss，标记 NA。

不得先逐样本计算恢复比例再简单平均。

同时报告：

```text
Gain_G2_vs_G1 = mean(loss_G1 - loss_G2)

Gain_G2_vs_G3 = mean(loss_G3 - loss_G2)
```

使用2000次固定种子的样本级配对 Bootstrap。

每次重采样重新计算组均值及比例指标。

若比例指标的分母在 Bootstrap 中非正，应报告无效重采样比例，不能据此生成误导性的稳定置信区间。



## v0.1.1 — 验证状态污染（R13）

在相同训练梯度和学习率 lr 下，评估积累 V_val 导致下一次手动更新相对于隔离评估多出 Δθ_contamination = lr * V_val。该式为本项目推导，依赖手动更新确实执行；实际污染规模未预设，不是原论文定理。v0.1.0 的 R1–R12 保持不变。


## v0.1.2：R14 辅助可恢复性上界

定义 A[b] 为 K_ref>0 且 K_alpha>0 的位置集合，I_abs[b]=sum_i,t |x_ref|。在 I_abs>0、非 eligible 位置系数固定为1的 R9/R10 下，E_floor[b]=sum_{i not in A[b]} |O_alpha-O_ref| / I_abs[b]。R_E_max=1-mean(E_floor)/mean(E_G1)，仅当 mean(E_G1)>0 且均值采用一致有效样本时定义。因非 eligible 项不能被改变，E_G2>=E_floor，故 R_E<=R_E_max；系数裁剪可能使实际恢复低于此上界。R14 是辅助指标，不更改原预注册门槛；不是任务损失或物理能量定理。


## v0.1.3：R15—R20

N=sum_i|O_alpha-O_ref|，I=sum_t,i|x_ref|，A=sum_i|O_ref|。R15: E_out=N/A；R16: q=A/I；R17: E=q*E_out，I和A均非零时逐样本成立，不能将样本均值拆乘。

R18: 正模块共享标量阈值且总量非零时，sum(O_alpha)/sum(O_ref)=alpha*(1-R_FR)，是总量等式而非逐位置等式。

R19: 对新增静默数N_silent，E_floor>=theta*N_silent/I；因此原补偿规则下 R_E<=1-mean(theta*N_silent/I)/mean(E_G1)，需有效正分母。

R20: 最后一层后仅GAP、线性读出且时间均值分类时，Delta_z=W*GAP(Delta_O)/T；不推广到存在下游非线性的前层。

clip后的逐位置最优系数 c_star 在eligible处取clip(O_ref/O_alpha,0.25,4)，非eligible取1；E_clip_min=sum_i|c_star O_alpha-O_ref|/I，R_E_attainable=1-mean(E_clip_min)/mean(E_G1)，仅有效正分母时成立。检查 R_E_attainable<=R14<=R19，所有上界仅限指定补偿规则。


## v0.1.4 — R21/R22：任务与计算成本感知候选

R21: 固定训练权重W，min L_task(W,theta)，subject to C(W,theta)<=C_target；C为本项目完整网络指定SOP口径。theta_l=alpha_l theta0_l，alpha只取固定阶梯。R22: delta_L=L(candidate)-L(current)，delta_c=[C(current)-C(candidate)]/C(v0)，score=delta_L/delta_c；只有 delta_c>=0.005 的动作有效，可有负delta_L。动态方法每个接受动作后重新测量全部可选单步动作，是贪心局部搜索，不保证全局最优；不预先宣称原创性。

## v0.1.5 — R23–R27 预算约束自蒸馏
R23：z=mean_t(logits)，CE=CE(z,y)，学生阈值固定为原阈值1.25倍；原AT-LIF动力学和替代梯度保留，关闭手动阈值更新。R24：KD=3² KL(softmax(z_teacher/3)||softmax(z_student/3))；教师冻结。R25：对实际被监控卷积，按输入位置的有效卷积扇出加权 ATLIF输出/固定阈值，按实际batch归一；多分支分别计数。该前向等价推导依赖正阈值及卷积输入确实来自ATLIF；反向为替代梯度，非硬阈值有限差分导数。R26：L_SOP=relu(C_proxy/54945117.504-0.8)²。R27：联合目标CE+0.5 KD+10 L_SOP。KD和活动正则化为成熟思路，性能恢复待验证。
