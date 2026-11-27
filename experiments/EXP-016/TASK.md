# Codex 科研执行任务：ROUND-015

# UniSparse 官方后端环境修复、工程准入及有效连接感知活动代价验证

## 一、任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-015
- 科研阶段：新Baseline阶段0，环境修复及理论机制准备
- 科研决策：DEC-015，REPAIR + CONTINUE
- 理论版本：v0.2.1
- 科学问题：Q-010，连接稀疏性与活动稀疏性的耦合优化
- 科研假设：H-016、候选H-017
- 实验编号：EXP-016
- 候选研究方向：ECAS（Effective-Connectivity-Aware Spike Sparsification）
- 项目仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-014-baseline-transition
- 本轮建议分支：research/round-015-unisparse-admission
- 上轮结果SHA：7f8b5b3901d4966e936d1da9f61a3bcc7012a19e
- UniSparse官方固定SHA：b5b3da4fcd2e7abcd2c20c93d41cf63ef3201daa
- 依赖实验：EXP-015
- 优先级：P0环境及官方后端正确性；P1完整模型资源测量；P1局部活动代价机制测试

首先读取AGENTS.md、RESEARCH_STATE.md、THEORY.md、HYPOTHESES.md、UNISPARSE_BASELINE.md、ROUND-014-TASK.md和EXP-015全部必要报告。

从ROUND-014最终结果SHA建立独立分支。

本轮禁止完整训练、正式性能优化、修改原始UniSparse算法及使用官方测试集。

## 二、ROUND-014正式科研判定

EXP-015为PARTIAL。

UniSparse准入为BLOCKED。

已经完成：

1. UniSparse官方固定版本导入。
2. 21个源码文件逐字节一致性核查。
3. MIT许可证记录。
4. 旧AT-LIF研究路线归档。
5. 论文—代码初步对应审查。
6. CPU合成权重稀疏及活动惩罚测试。
7. 有效连接掩码与稀疏卷积输出检查。
8. 二值脉冲SOPMonitor与独立计数对照。
9. 训练数据隔离入口。
10. R54/R55结构—活动成本分解的合成测试。

上轮SOP合成实例：

- 原始连接与原始活动：8。
- 稀疏连接与原始活动：6。
- 稀疏连接与改变后活动：3。
- 结构项：-2。
- 活动项：-3。
- 总变化：-5。

这些是UNIT_TEST，不是正式模型性能。

未完成：

- 官方CuPy后端运行。
- 官方GPU前向和反向验证。
- 完整模型稀疏机制正确性检查。
- GPU资源测量。
- 300轮训练成本估计。
- 正式非零lambda配置确认。

实际GPU使用0秒。

上轮主要阻塞是缺少CuPy，不是已确认的算法失败。

不需要重复已经完成的全部CPU测试，只需必要的回归测试。

## 三、本轮研究目标

1. 在不破坏既有AT-LIF实验环境的情况下解决CuPy缺失。
2. 验证官方CuPy神经元与参考Torch实现的前向、反向及重置行为。
3. 完成UniSparse CifarNet单GPU正确性检查。
4. 获取真实训练批次时间、显存和稀疏惩罚额外开销。
5. 验证名义连接扇出与有效连接扇出在局部活动代价上的区别。
6. 判断UniSparse是否满足下一轮完整Baseline训练准入条件。

本轮不要求获得准确率提升。

## 四、固定官方版本及源代码保护

继续使用：

vendor/UniSparse/

官方仓库：

https://github.com/xyshi2000/UniSparse

官方固定版本：

b5b3da4fcd2e7abcd2c20c93d41cf63ef3201daa

不直接修改该目录下的原始官方文件。

新环境适配、后端检查、受控入口及理论测量工具优先放置在：

experiments/ROUND-015/

或根据实际仓库规范建立等价独立模块。

所有必要兼容补丁必须：

- 记录修改前后的内容。
- 记录目标文件哈希。
- 记录修改理由。
- 检查是否影响数学语义。
- 不改变既有AT-LIF代码和运行环境。

## 五、CuPy独立环境修复

上轮实际环境：

- Python 3.11.9
- PyTorch 2.4.1+cu118
- torchvision 0.19.1+cu118
- NumPy 1.26.4
- SpikingJelly 0.0.0.0.14
- timm 1.0.22
- NVIDIA Driver 550.144.03
- nvcc 11.8
- CuPy未安装

上轮虚拟环境：

.local/round014-venv

该环境继承系统包，并非完全独立依赖闭包。

本轮优先继续使用该专用虚拟环境，或者在不增加大量依赖下载的条件下建立更独立的UniSparse虚拟环境。

首先核查：

- sys.executable。
- sys.prefix及sys.base_prefix。
- pip实际安装目标。
- 已有核心依赖的真实文件路径。
- 磁盘空间及下载权限。
- CUDA Toolkit真实安装位置。
- 现有CuPy包是否出现多版本冲突。

推荐首先尝试：

cupy-cuda11x==13.5.1

依据：

https://docs.cupy.dev/en/v13.5.1/install.html

https://pypi.org/project/cupy-cuda11x/13.5.1/

该版本支持CUDA 11.2至11.8，并提供Python 3.11 Linux预编译wheel。

此版本为本轮建议环境适配选择，不是官方论文已经确认使用的确切版本。

先使用pip dry-run或等价机制检查安装计划。

确认不会升级或卸载既有PyTorch、torchvision、NumPy、SpikingJelly等核心包。

优先使用预编译wheel，不从源码编译CuPy。

示例：

.local/round014-venv/bin/python -m pip install --only-binary=:all: "cupy-cuda11x==13.5.1"

具体命令以实际环境核查为准。

本轮建议新增下载总量上限200 MB，新增依赖环境磁盘占用上限2 GiB。

超过上限停止并报告，不自动下载另一套完整CUDA或PyTorch运行环境。

不得：

- 使用sudo修改系统环境。
- 修改显卡驱动。
- 升级系统CUDA Toolkit。
- 修改旧AT-LIF环境。
- 同时安装多个不同CUDA系列的CuPy包。
- 在无法确认安装目标时执行pip安装。

安装后记录：

- pip freeze。
- CuPy实际版本。
- cupy.show_config()。
- CuPy CUDA Runtime及Driver版本。
- 实际动态库路径。
- Python模块路径。
- pip安装日志及wheel来源。

运行前后核查旧环境关键依赖版本与源文件完整性。

如果13.5.1无法安装或无法加载，不自动进行大量版本搜索。

记录真实报错并结束相应任务，等待理论研究员和用户决定后续兼容方案。

## 六、SpikingJelly兼容问题处理

UniSparse官方README提到：

spikingjelly==0.0.0.0.14

与较新NumPy版本可能存在np.int兼容问题。

必须先检查实际运行时是否触发相关报错。

不得为了消除潜在问题提前修改不相关代码。

若需要执行官方README建议的np.int到np.int_兼容补丁：

1. 确认目标文件位置。
2. 确认目标位于UniSparse专用环境内部。
3. 若目标文件来自旧环境继承包，禁止直接修改。
4. 将对应版本安装或复制到专用环境自身包目录。
5. 保持原SpikingJelly版本不变。
6. 只修改必要兼容行。
7. 保存补丁和哈希。
8. 重新运行必要正确性检查。

如果需要修改神经元动力学、梯度公式或CUDA kernel的数学逻辑，停止并提交理论确认，不擅自修复。

## 七、EXP-016A：CuPy环境冒烟测试

GPU测试前必须确认：

- 仅使用一张符合实验室共享GPU管理规定的RTX4090。
- CUDA_VISIBLE_DEVICES正确隔离。
- PyTorch和CuPy均识别相同物理GPU。
- 剩余显存与计算负载满足本轮测试要求。

不得终止、暂停或干扰其他用户的GPU进程。

执行：

1. CuPy数组创建和基础运算。
2. CuPy GPU kernel实际执行。
3. PyTorch GPU tensor运算。
4. PyTorch与CuPy之间必要的设备互操作。
5. CUDA同步及错误检查。
6. 官方LIF神经元实例化。
7. FP32官方LIF前向及反向。

CuPy成功import不等于CUDA kernel成功执行。

必须获得真实GPU计算结果。

记录首次编译成本及后续缓存复用成本。

## 八、EXP-016B：神经元后端正确性对照

依据官方实现：

vendor/UniSparse/models/submodules/layers.py

原LIF：

- tau=2。
- decay_input=True。
- v_threshold=1。
- v_reset=0。
- detach_reset=True。
- surrogate_function=ATan()。
- step_mode=m。
- backend=cupy。

使用同参数的Torch GPU后端作为参考。

必要时增加Torch CPU float64参考。

不将参考后端作为未经验证的正式替代实现。

固定小型人工输入序列，覆盖：

- 从不发放。
- 持续发放。
- 时间累积后发放。
- 发放后硬重置。
- 正负输入。
- 接近阈值的输入。
- 多时间步及多通道。
- 不同上游梯度。
- 正常混合精度。
- 非有限梯度及AMP跳步情形。

在可能的情况下直接比较原始膜电位序列；如果需要额外开启store_v_seq，必须明确这是诊断设置。

检查：

- 二值脉冲序列。
- 最终及逐时刻膜电位。
- 输入代理梯度。
- 硬重置结果。
- reset_net后的状态。
- 连续运行前后状态隔离。

FP32建议容差：

max(abs(actual-reference)/(1+abs(reference))) <= 1e-5。

混合精度建议容差1e-3。

对非阈值边界样本要求脉冲完全一致。

接近阈值的样本可能受浮点舍入影响，需要独立记录真实分歧及膜电位与阈值之间的距离，不以放宽容差掩盖问题。

硬阈值函数有限差分不能作为ATan替代梯度的严格标准。

禁止擅自调整tau、阈值、重置方式或替代梯度。

## 九、完整模型受控入口

本轮固定：

- UniSparse官方CifarNet。
- CIFAR-10。
- T=4。
- base_channels=256。
- Batch Size=256。
- seed=12450。
- 官方SGD参数组。
- 官方训练数据增强。
- 官方AMP协议。
- 梯度累积步数按实际官方配置核对，预期为1。

以上参数来源于上轮代码审查，不能替代对实际解析参数的核对。

保留原官方网络及神经元实现。

复用ROUND-014训练来源数据入口。

固定数据划分：

experiments/EXP-004/splits/indices.json

SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

只读取CIFAR-10训练数据文件。

禁止构造、读取或评价官方测试集。

禁止直接调用会自动使用官方测试集选最佳模型的原main.py完整训练入口。

准备独立受控训练批次入口，保持数学更新机制与原代码一致。

本轮不运行完整训练Epoch，也不进行正式4500张验证集性能评价。

## 十、完整模型稀疏机制检查

使用随机初始化、明确标记为SMOKE_TEST的CifarNet。

首先完成必要的单步检查：

1. 原始前向输出有限。
2. 输出时间及类别维度正确。
3. 原始任务损失有限。
4. 梯度有限。
5. 实际优化器参数组正确。
6. 权重阈值是buffer还是Parameter。
7. BN gamma、beta是否参与任务优化。
8. sparse_weight与sparse_mask一致。
9. 权重阈值为零时前向恢复原始权重。
10. 非零阈值是否造成实际有效连接变化。
11. calc_penalty是否成功执行。
12. local_forward是否正确重置神经元状态。
13. 临时梯度保存与恢复是否正确。
14. 惩罚buffer能否正确清除。
15. 手动惩罚更新是否实际生效。
16. 原作者Monitor在完整模型中是否可正常工作。

记录执行惩罚前后的参数及buffer变化。

不能将非零惩罚值等同于实际有效剪枝。

额外检查原main.py的更新顺序：

task forward/backward

functional.reset_net(model)

optimizer/scaler step

optimizer.zero_grad()

calc_penalty(model)

penalize(model)

clear_penalty(model)

该顺序意味着局部惩罚可能使用缓存的旧前向输入及优化器更新后的BN affine参数。

将其作为真实代码机制记录。

本轮不将该顺序认定为错误，不擅自交换更新步骤。

## 十一、AMP跳步协议检查

ROUND-014已发现：

在CPU人工溢出场景中，按照原调用顺序，任务优化器跳步后手动权重阈值仍可能改变。

本轮在小型GPU合成场景中检查：

- AMP是否实际跳过optimizer step。
- threshold buffer是否仍然变化。
- BN gamma/beta手动惩罚是否仍然执行。
- 其他可训练参数是否变化。
- 惩罚状态是否被正确清理。
- 是否存在非有限手动更新。

只记录原作者行为。

本轮保留原始更新顺序，不擅自决定将所有手动更新与AMP成功步绑定。

如发现非有限值或不可接受的状态污染，停止正式资源测量并报告。

## 十二、理论研究员新增发现

实际源码：

vendor/UniSparse/utils/utils.py

calc_penalty中的活动惩罚使用：

weight = conv.out_channels * conv.kernel_size[0] * conv.kernel_size[1]

随后：

spike = spike.mean(dim=(0,1,3,4)).sum(0) * next_weight

这里没有直接读取后继卷积的sparse_mask。

而原SOPMonitor使用：

conv2d(x, m.sparse_mask(), ...).sum()

因此，训练活动代理使用名义连接扇出，实际SOP评价使用有效连接。

该差异由真实代码证实。

不预先认定它会降低任务性能。

新增理论R57—R61。

### R57：原活动惩罚

对于单个直接后继卷积：

r[c] = sum(s[t,b,c,h,w]) / (T*B*H*W)

P_original = C_out*k_h*k_w*sum_c(r[c])

多个后继卷积时，原实现累加各自名义扇出系数。

### R58：有效扇出

d[c] = sum_{j,a,b} M[j,c,a,b]

M为实际有效权重二值掩码。

### R59：通道级近似有效活动代价

P_effective = sum_c(d[c]*r[c])

适用条件：

- 直接卷积消费关系。
- 忽略边界效应。
- stride=1。
- 无中间池化或其他非线性变换。
- 实际张量通道正确对应。

稠密连接条件下恢复R57。

R59不是所有真实网络结构下的精确SOP公式。

### R60：精确空间有效扇出

F[c,h,w]为该输入位置到全部实际输出位置之间的有效连接数。

C(M,s) = sum_{t,b,c,h,w}(F[c,h,w]*s[t,b,c,h,w])/B

F必须考虑：

- 实际输入输出形状。
- stride。
- padding。
- dilation。
- groups。
- 真实有效连接掩码。

可以通过显式连接枚举获得参考值，也可以使用经过独立验证的卷积转置或对应输入梯度运算得到。

### R61：固定连接时的局部代价

dC/ds[t,b,c,h,w] = F[c,h,w]/B

这里将s连续化，仅用于局部计算代价的数学表达。

不宣称原二值发放函数可微。

不包含脉冲变化对后续其他层产生的间接影响。

本轮仅测试上述公式，不将其加入正式训练。

## 十三、EXP-016D：有效连接感知活动代价测试

使用CPU合成小型卷积优先完成。

### 13.1 通道有效连接差异

构造两个输入通道：

- 相同名义连接规模。
- 相同发放率。
- 一个通道保留9条有效连接。
- 另一个通道保留1条有效连接。

选择无边界干扰的简单卷积或等价连接配置。

比较：

- 原名义扇出惩罚。
- 有效连接扇出惩罚。
- 删除一个指定脉冲后真实SOP变化。

预期在适用条件下，原惩罚对两个通道的系数相同，真实SOP边际代价分别为9和1。

实际结果不一致时，先核查张量布局、归一化和连接定义，不强行修改结果。

### 13.2 稠密及稀疏掩码

固定二值输入活动。

构造预先确定的掩码：

- 全稠密。
- 均匀稀疏。
- 通道间非均匀稀疏。
- 某通道全部连接删除。
- 全部连接删除。

分别记录：

- 实际权重密度。
- 原活动惩罚。
- 通道有效扇出。
- 精确位置有效扇出。
- 作者SOPMonitor。
- 独立逐连接计数。
- R60结果。

在可比口径下验证掩码变化是否改变真实代价但不改变原名义扇出系数。

### 13.3 空间边界及步幅测试

固定少量确定性案例，覆盖：

- 1x1卷积。
- 3x3卷积，padding=1。
- stride=2。
- dilation变化。
- 非方形输入。
- 不完整Batch。

groups>1不是当前SparseConv2d支持范围，只允许作为独立计数公式测试，不能宣称UniSparse支持分组稀疏卷积。

用显式枚举验证R60。

R59与真实SOPs的边界误差必须独立记录，不能把近似公式当作严格等式。

### 13.4 实际CifarNet图审查

读取：

vendor/UniSparse/models/cifarnet.py

vendor/UniSparse/models/submodules/blocks.py

实际记录SparseNeuron与后继卷积的注册关系。

重点检查：

- ConvBlock之间直接消费关系。
- activation2之后的MaxPooling。
- activation5之后的MaxPooling与flatten。
- fc1的1x1稀疏卷积。
- IdentityBN是否具有可惩罚的affine参数。
- 静态首层及SOPMonitor跳过范围。

不能直接对存在池化或flatten的模块应用未修正的通道扇出公式。

优先选择具有直接消费关系的ConvBlock路径进行真实网络测量。

在随机初始化的完整模型中，可以使用独立的合成权重掩码干预测量局部代价变化。

任何人工掩码不得保存成正式Baseline模型。

该结果只能用于验证H-016的机制及测量工具，不能说明训练后网络中确实存在具有实际科研意义的代价失配。

## 十四、SOP统计口径修正准备

原作者main.py使用批次SOP记录均值除以配置batch_size。

对于尾Batch不足的情况，需额外检查：

C_exact = sum(所有批次SOP总数) / sum(所有批次实际样本数)

与原作者聚合方式是否存在差异。

保存两个口径：

- SOP_author。
- SOP_sample_weighted。

不静默覆盖作者定义。

如果两者出现差异，报告原因、适用范围及数值。

对于非二值输入，原Monitor直接求和并不等同于严格的二值事件计数。

本轮以CifarNet真实输入类型核对适用条件。

不把理论运算量解释成硬件实测能耗或GPU推理加速。

## 十五、EXP-016C：单GPU资源测量

所有必要的环境、后端及完整模型正确性检查通过后执行。

固定CifarNet、T=4、batch_size=256。

最多40个训练批次级资源测量步骤，包含：

- 最多10个预热Batch。
- 最多30个正式测量Batch。

前置正确性测试使用的额外少量单步操作需单独计入预算，不混入正式测速。

建议在总测量批次限制内分别覆盖：

A：零稀疏惩罚路径。

B：非零稀疏惩罚路径。

每种路径从相同初始化独立开始。

优先使用可以从原论文或正式补充材料确认的单个代表性lambda配置。

如果无法确认论文正式非零lambda：

允许使用已明确标记的合成工程测试系数0.01，目的仅为让权重和活动惩罚代码被执行，测量额外开销。

明确说明：

- 该系数不是正式论文超参数。
- 不用于选择稀疏率工作点。
- 不用于评价分类性能。
- 非零路径的速度仅代表当前测试条件。
- 正式Baseline的lambda仍需单独确认。

不进行lambda搜索。

记录：

- 数据加载耗时。
- 任务前向耗时。
- 任务反向耗时。
- optimizer耗时。
- 局部惩罚重新前向耗时。
- 局部惩罚反向耗时。
- 手动更新及清理耗时。
- 单批次整体墙钟时间。
- CUDA kernel首次编译时间。
- 峰值allocated显存。
- 峰值reserved显存。
- AMP跳步次数。
- 成功优化器更新次数。
- 实际执行的惩罚更新次数。

若进行CUDA event或强制同步分阶段计时，报告测量本身对整体吞吐的影响。

不直接把同步分阶段计时之和视为无额外开销的训练速度。

## 十六、训练成本估计

基于本轮真实非零惩罚路径测量结果，计算：

每轮训练成本 = 实测平均批次耗时 × 实际训练批次数。

300轮训练成本 = 每轮训练成本 × 300。

训练批次数根据45000样本、实际Batch Size和drop_last规则确定。

分别报告：

- 零惩罚路径估计。
- 非零惩罚工程路径估计。
- 首次CUDA编译成本。
- 验证及检查点保存额外成本。
- 共享GPU负载波动。
- 随机初始化短程测速的适用边界。

如果非零lambda仍不是正式论文配置，则正式Baseline训练成本只能标记为工程估算，不能宣称精确。

若资源测量未完成，不利用旧AT-LIF训练速度填补缺失数据。

## 十七、科研假设与判定

H-016：

UniSparse原活动惩罚使用固定名义扇出，在实际有效连接变化时可能与局部真实SOP代价不一致。

本轮最低机制验收：

1. 原活动惩罚公式与实际代码一致。
2. 固定脉冲、改变掩码时，原名义系数保持不变。
3. 真实SOPs随有效连接变化而改变。
4. R60与独立连接计数一致。
5. R61能够正确描述固定连接下单个脉冲的SOP代价。
6. 稠密或满足适用条件时能够恢复相应原始表达式。

以上条件通过，只标记为合成机制及数学实现得到支持。

H-016在正常训练网络中的实际影响仍待验证。

H-017：

有效连接感知活动惩罚可能改善同SOPs预算下的任务性能。

本轮保持UNTESTED。

不得根据CPU合成测试、随机初始化或短程测速结果宣称H-017成立。

## 十八、UniSparse准入门槛

必须满足：

1. 官方固定版本保持可追溯。
2. 专用环境兼容。
3. 旧环境未被破坏。
4. 官方CuPy LIF前向和反向通过。
5. 主要Torch/CuPy后端一致性检查通过。
6. CifarNet完整前向和反向成功。
7. 非零稀疏惩罚路径可执行。
8. 权重掩码及实际更新状态正确。
9. SOP计数口径清楚。
10. 官方测试集保持隔离。
11. 真实GPU资源测量完成。
12. 关键未解决理论及配置问题被明确记录。

若全部工程门槛通过，记录：

ENGINEERING_ADMITTED

但如果正式论文lambda、复现实验配置或理论审查仍存在关键不确定性，正式完整训练批准状态保持PENDING。

不自动开始300轮训练。

## 十九、资源预算及异常停止

本轮需要用户新授权。

建议：

- 单张符合实验室共享管理规定的RTX4090。
- GPU总占用上限15分钟。
- CPU累计计算上限45分钟。
- 新增依赖下载上限200 MB。
- 新增依赖环境磁盘占用上限2 GiB。
- 正式完整训练0 Epoch。

CPU累计计算及GPU实际占用分别记录。

网络下载等待、解释器启动、环境准备及GPU等待分别记录，不混为CPU计算时间。

若发生：

- 依赖安装会破坏旧环境。
- CuPy无法编译或执行实际kernel。
- 官方神经元行为检查失败。
- CUDA资源不足。
- 数据隔离失败。
- 非有限状态。
- 超出批准预算。

停止相应实验。

不得自动扩大安装范围、尝试大量版本、增加训练轮数、切换研究Baseline或启动未授权后台任务。

## 二十、论文与创新性审查记录

在LITERATURE.md追加：

1. UniSparse: Combining Weight Pruning and Spike Sparsification in Spiking Neural Networks。

https://proceedings.mlr.press/v306/shi26a.html

2. Optimizing the Energy Consumption of Spiking Neural Networks for Neuromorphic Applications，2020。

https://doi.org/10.3389/fnins.2020.00662

3. Backpropagation With Sparsity Regularization for Spiking Neural Network Learning，2022。

https://doi.org/10.3389/fnins.2022.760298

4. Towards Energy Efficient Spiking Neural Networks: An Unstructured Pruning Framework，ICLR 2024。

记录已有研究已经涉及SynOps正则化、权重及脉冲联合稀疏。

不宣称有效连接感知活动惩罚为学术首创。

补充保存原论文公式(6)—(8)、(14)—(19)、(20)—(32)及Algorithm 1的具体文字或数学表达与页码定位。

重点确认：

- 原论文究竟如何定义有效连接依赖。
- 原作者是否明确说明了名义扇出近似。
- 论文联合稀疏目标与实际局部惩罚的关系。
- 是否能从论文正文、附录及正式公开记录确认CifarNet的代表性非零lambda。

理论研究员本轮独立访问全文PDF受到限制。

Codex应保留原论文可核实链接、哈希、相关页码及公式证据，不编造缺失内容。

## 二十一、科研状态更新

新增：

research/ROUND-015-TASK.md

更新：

- RESEARCH_STATE.md。
- UNISPARSE_BASELINE.md。
- THEORY.md：追加v0.2.1及R57—R61。
- HYPOTHESES.md：新增H-016及候选H-017。
- EXPERIMENT_PLAN.md。
- EXPERIMENT_REGISTRY.csv。
- DECISION_LOG.md：追加DEC-015。
- LITERATURE.md。

保留原AT-LIF全部历史科研状态。

EXP-015保持PARTIAL，不因后续环境修复而覆盖历史执行结果。

EXP-016按真实执行结果登记。

旧阻塞解决后，在新实验记录中建立关联，不重写原失败记录。

## 二十二、实验交付

建立experiments/EXP-016独立目录及唯一Run ID。

至少保存：

- TASK.md
- config.json
- environment_before.txt
- environment_after.txt
- dependency_install.log
- dependency_lock.txt
- compatibility_patch.diff（如有）
- legacy_environment_integrity.json
- upstream_version.json
- code_version.json
- commands.sh
- status.json
- cupy_runtime_tests.json
- neuron_backend_comparison.json
- mixed_precision_tests.json
- complete_model_smoke.json
- sparse_mechanism_tests.json
- amp_update_order_tests.json
- connectivity_cost_tests.json
- connectivity_cost_samples.csv
- consumer_dependency_map.json
- sop_aggregation_audit.json
- resource_benchmark.csv
- resource_estimate.json
- baseline_admission.json
- run.log
- REPORT.md
- limitations.md
- figures/

未执行项目记录NA及实际原因。

无实际数据时不生成虚构图表或结果文件。

本轮可以保存必要的小型合成输入及参考张量，不上传大型检查点、完整数据集或依赖wheel。

## 二十三、GitHub最终交付

正式实验执行前提交任务、固定配置和相关实现。

记录每次实际运行的代码SHA。

实验结束后同步全部真实结果和科研状态。

不修改main，不force push，不覆盖历史实验。

最终汇报：

- GitHub实验分支链接。
- 执行代码SHA。
- 最终结果提交SHA。
- EXP-016执行状态。
- CuPy安装结果及实际版本。
- 是否影响旧环境。
- 官方CuPy神经元正确性结果。
- CifarNet完整前向、反向及非零惩罚结果。
- AMP跳步及手动更新观察。
- R57—R61机制测试结果。
- 原名义扇出与实际有效扇出对照。
- 实际单批次时间和显存。
- 零惩罚及非零惩罚路径开销。
- 300轮训练成本工程估计。
- 正式lambda是否确认。
- UniSparse准入结论。
- 全部异常、阻塞及局限。
- 报告和原始数据路径。

本轮只解决新Baseline的工程准入和理论测量基础。

禁止提前开展ECAS正式方法训练。

获得用户授权后按固定协议执行。 