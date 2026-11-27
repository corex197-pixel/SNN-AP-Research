# UniSparse 论文与工程审查（Q-010）

Shi, Bu & Yu, *UniSparse: Combining Weight Pruning and Spike Sparsification in Spiking Neural Networks*, ICML 2026, PMLR 306:111711–111725. [论文主页](https://proceedings.mlr.press/v306/shi26a.html)，[全文与附录](https://raw.githubusercontent.com/mlresearch/v306/main/assets/shi26a/shi26a.pdf)，[官方源码](https://github.com/xyshi2000/UniSparse/tree/b5b3da4fcd2e7abcd2c20c93d41cf63ef3201daa)。全文 PDF 成功读取，哈希见 EXP-015/upstream_version.json；不将论文声称的结果当成项目复现。

## 论文原编号及代码对应

| 论文公式/位置 | 论文内容 | 官方固定源码的位置和实际行为 |
|---|---|---|
| (1) | LIF 充电、二值发放、硬重置；τ=2，阈值1，rest=0 | `vendor/UniSparse/models/submodules/layers.py:16-20`：SpikingJelly LIF，`decay_input=True`、`detach_reset=True`、ATan、CuPy。 |
| (2)–(3) | BN 标准化、仿射及与线性层融合 | `layers.py:23-26,83-111`；BN γ/β 是可训练参数。融合是论文推导，不表示代码训练时执行融合。 |
| (4)–(8) | SOP 能源代理，联合脉冲与有效连接能量目标 | `utils/utils.py:419-477` 统计 sparse_mask 与输入；`utils/utils.py:28-78` 构造局部惩罚。论文结构映射与代码近似是否完全一致仍待理论审查。 |
| (9)–(13)、Theorem 4.1 | BN 前输入与时间累积、有效发放条件及单调性；结论有 γ>0 等假设 | `layers.py:100-110,147-155` 重建局部 BN/LIF 前向；论文有效阈值关系是推导，不是额外直接可训练神经元阈值。 |
| (14)–(19) | 局部脉冲损失与 γ/β 梯度更新；论文聚合定义及 ATan 代理梯度 | `utils/utils.py:28-78` 平均时间、样本、空间并对通道求和；后继权重因子由 `out_channels*kernel_area` 计算；`layers.py:90-110` 手动更新 γ/β。该系数并未在此处读取后继 `sparse_mask`，依赖解释需理论审查。 |
| (20) | `sign(θ) ReLU(|θ|-d)` 软阈值权重 | `layers.py:42-64,73-80`；threshold 是非训练 buffer，mask 来自真实软阈值结果，正向为稠密张量卷积。 |
| (21)–(27) | 权重稀疏目标、L1 松弛、θ 与阈值更新、被剪权重代理梯度 | `layers.py:29-39,51-52,66-71`；PseudoRelu 反向直通、阈值手动递增。确切 θ=0 边界仍值得审查。 |
| (28)–(32)、附录 Algorithm 1 | 权重和脉冲的联合约束及四项更新 | `main.py:391-434,781-788`、`utils/utils.py:28-86`。代码接受两个 lambda，训练时乘当前 lr；论文 Algorithm 1 描述统一 λ。关系与稀疏依赖须审查。 |
| 附录 C | CIFAR-10 三模型，SGD momentum 0.9、lr 0.1 cosine、300 epoch、batch256、T4、三种子 | `configs/cifar10_cifarnet.yaml:1-2`、`main.py:50-54,88-107,122-136,692-725`；当前 timm `create_optimizer_v2` 的 momentum 默认0.9，参数组实际构造尚需完整模型环境验证。论文给的多个稀疏结果未核定单个非零 λ，正式非零资源测量不应猜测。 |

`pre_norm_input` 训练时 `clone().detach()`；局部 BN 前向使用当前批统计而不写运行统计，局部反向暂存/恢复任务梯度，惩罚 buffer 经 `penalize` 后清理。具体频率与梯度累积见 `main.py:411-434`。官方 AMP `scaler.step/update` 后未检查是否跳步就继续手动惩罚；CPU 合成溢出示例证实这一调用顺序可能出现优化器跳过而 threshold 改变，但尚非 CuPy GPU 验证，不能据此下算法缺陷结论。

稀疏卷积 mask 表示非零有效连接；总参数量、非零权重、活动率、理论 SOP、张量尺寸、耗时须分开。作者 monitor 对 SparseConv2d 使用 sparse_mask；普通 Conv2d 使用全 1 核，跳过 `net.skip`；输入直接求和而非强制二值化。CifarNet 跳过 `static_conv`。`main.py:503-522` 对 batch 记录取均值并按配置 batch_size 归一，末批不足时存在口径差异的可能；不能默认为按真实样本精确加权。

完整机制测试、数据入口差异和可追溯评估结果见 `experiments/EXP-015/`。下一步理论审查重点：公式(6)–(8)的映射与代码形状系数、(17)–(19)的时空聚合、(20)–(27)代理梯度边界、(28)–(32)双 lambda 与实际连接依赖、AMP 跳步时手动更新语义。官方测试集及模型选择必须隔离。

## ROUND-015：论文原式定位与已明示近似

[论文全文 PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/shi26a/shi26a.pdf) 的 SHA256 见 `experiments/EXP-016/upstream_version.json`。下面为核对用公式概述；符号和假设以论文为准。

- PDF 第4页，(6) `E=Σ_t Σ_i s_i[t] Σ_j 1{w_ij≠0}`；(7) `E(s)=||s⊙g_s(s)||₁`；(8) `E(w)=||g_w(w)⊙1{w≠0}||₁`。这里的映射在原始目标中描述脉冲与有效连接的耦合。
- 第5页，(14) 为任务损失与活动能量正则联合最小化；(15)–(16) 分别为 γ、β 的任务梯度加能量梯度更新；(17) 的局部代理导数分别为 `Θ′(v−vth) ẑ` 与 `Θ′(v−vth) D`；(18)–(19) 将其与连接映射及共享参数聚合 h 结合。
- 第5–6页，(20) `w(θ,d)=sign(θ) ReLU(|θ|−d)`；(21) 为权重 L0 能量目标，(22) 为 L1 松弛；(23) 为松弛后的 w 梯度更新，(24) 为 θ 链式更新，(25) 为 `d←d+ληg_w(w)`；(26)–(27) 使用剪除区间的代理梯度更新 θ 和 d。
- 第6–7页，(28) 是 `min L(f(w(θ,d),γ,β))+λE(w(θ,d),s)`；(29)–(32) 依次更新 θ、d、γ、β。第7页正文明确解释：为缓解可能的正反馈，使用描述**初始网络结构**的常数近似 `g_w` 和 `g_s`。第12–13页附录 B / Algorithm 1 给出实际更新顺序。名义扇出不随有效掩码变化是作者已说明的近似，不能表述为本项目首次发现或直接认定为实现错误。

`calc_penalty` 实际使用 `C_out*k_h*k_w`，而 `SOPMonitor` 使用 `sparse_mask`，合成实验验证两者在固定活动而有效连接变化时可给出不同局部代价。实验未证明训练后模型中差异具有实际性能影响。CifarNet 直接路径、池化与 flatten/IdentityBN 的注册关系见 `experiments/EXP-016/consumer_dependency_map.json`。正式非零 λ 的单一代表值尚未在正文、附录或固定代码配置中验证；本轮 0.01 仅为工程测试。

权重部分更精确的公式索引（第6–7页，令 w=w(θ,d)，L 为任务损失）：(21) `min_θ L+λ||g_w(w)⊙1{w≠0}||₁`；(22) `min_θ L+λ||g_w(w)⊙w||₁`；(23) `w←w−η∇_w L−ληg_w(w)⊙sign(w)`；(24) `θ←θ−η∇_w L⊙∇_θw`；(25) `d←d+ληg_w(w)`；(26) `θ←θ−η∇_w L`；(27) 与 (25) 同式；(29)–(32) 依次为 `θ_l←θ_l−η∇_{w_l}L`、`d_l←d_l+ληg_w^l`、`γ_l←γ_l−η∇_{γ_l}L−ληg_s^l h(Θ′_l ẑ_l)`、`β_l←β_l−η∇_{β_l}L−ληg_s^l h(Θ′_l D_l)`。`g_s^l,g_w^l` 是作者规定的初始结构常数近似，不是实时计算的有效 mask。
