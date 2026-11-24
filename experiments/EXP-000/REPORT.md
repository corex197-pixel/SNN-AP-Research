# EXP-000 Baseline 工程审查 — COMPLETED（权重可用性受限）

基线为本项目导入提交 `52878cff06f6378fa155c032b7acd23883a72e4a`；这不是上游 commit。上游 HEAD 查询为 `01c481001e3f21d1b924a85aedd8126e4cfeb05d`，未证明本地与其一致。未找到可追溯的本项目训练记录或权重；不声称完整复现。环境见 environment.txt；`main` 与 ATLIF 的 Python 导入成功。`/home/hym/dataset/cifar10/cifar-10-batches-py/` 存在 CIFAR-10 批文件，未检查校验与训练数据暴露历史。在 `/home/hym` 排除 dataset 进行文件名搜索未发现 .pt/.pth/.ckpt；此搜索不排除其他位置或其他扩展名的权重。dataset 内 .pt 不可视为模型权重。

## 公式—代码表（路径均相对项目根目录，行号为 baseline 导入版）

| 项 | 审查 |
|---|---|
| 有效输入、充电 | `Activity-Pruning-SNN-main/models/submodules/layers.py:90-95`：输入 `x[t,...]`，`mem=tau*mem+x`；CIFAR 主要建模入口 `main.py:645-651` 传 tau=0.9。|
| 边界/输出/复位 | `layers.py:61-65,90-99`：`input >= thre`；输出 `s*thre`，`mem=(1-spike/thresh.detach())*mem` 前向硬归零。论文描述达到阈值即发放；≥ 与之相符。|
| 状态和共享 | `layers.py:77-80,90-103`：每模块一个标量可学习阈值；每次 forward `mem=0`，跨调用无膜电位延续；时间优先 [T,...]。`main.py:434,474` 还调用 reset_net。|
| 膜电位梯度 | `layers.py:68-73` 对上游梯度乘一次 `g=max(1-abs((m-theta)/theta),0)`；这里是替代梯度，不是真实导数。|
| 阈值梯度差异 | `layers.py:72-74` 先令 grad_input=upstream*g，再 `grad_thre=-(grad_input*g).mean()`，即 `-mean(upstream*g²)`，不是 Eq.13 的逐元素 `-g`；张量先按所有维度求均值。|
| 复位 detach | `layers.py:96` 只 detach 分母阈值；`spike` 未 detach，复位路径仍保留替代梯度。|
| 手动阈值更新 | `layers.py:63,98` 每步对 `sp*g*s` 先沿第 0 维求和、其余求均值，再除 T 跨步累加；若输入为 [B,C,H,W]，第0维为 batch，故对 batch 求和、对其余维度求平均（标量输入形状需另论）。`utils/utils.py:413-418` 优化器之后额外加 lr*update_value 并清零；`main.py:668-673,419-427` 优化器接收 model，阈值是参数。无正阈值约束已发现；实际适用前需验证检查点阈值为正。|
| 活动正则 | `layers.py:99-102` 对缩放输出按每时刻 batch 与空间等维度取均值后跨时间求和；`utils/utils.py:406-410` 跨模块求和，`main.py:402` 乘 eta2。不是直接二值脉冲正则。|
| FR / SOP | `layers.py:100-101` 模块 FR 为所有时间/样本/空间 `output/theta` 的均值，count 是 batch 均值后求和；`main.py:477-495` 对 batch FR 再算平均。`utils/utils.py:421-449` 将卷积输入 `>0` 二值化、用全 1 权重卷积、排除 `net.skip`；并未纳入权重是否为零，不能称有效稀疏连接计数；`main.py:565-575` 再按 batch size 与步模式缩放。代码打印的 Power 为系数估算，非硬件实测。|
| 入口 | `main.py:174-185` 加载 CIFAR-10 训练/测试数据；`train_one_epoch:366`；`evaluate:460`；`test:502`；`main:578`；恢复训练 `713-719`，测试读取 `853-860`。本代码未发现独立可执行永久权重剪枝入口，不排除其他文件。|

默认配置 `cifar10.yaml` 是 T=8、ResNet20、eta=0.001、eta2=0.0001，不等于任务优先的 eta=0.002、eta2=0；data_path 为占位符。当前数据可见不证明预先独立验证集可用或模型训练数据暴露历史已知。

## 适用范围及后续
EXP-001 验证离散硬重置前向恒等式；真实网络需确认全部阈值正、输入固定、模块无重复/反馈、状态捕获无副作用。EXP-002 缺可靠权重及状态捕获核验，BLOCKED；EXP-003 的门槛不可评估。理论审阅者需判断梯度与 Eq.13 差异对训练解释的影响；本轮未修改 baseline。
