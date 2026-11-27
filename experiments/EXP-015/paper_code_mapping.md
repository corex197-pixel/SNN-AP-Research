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
