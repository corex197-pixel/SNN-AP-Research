# ROUND-013 候选 Baseline 静态可复现性审查

核查方式：2026-11-27 阅读公开代码仓库、README、代码入口及论文；未安装依赖、下载大数据或运行训练。UNKNOWN 表示未在所核查的来源中确认，而非断言不存在。

| 项目 | 代码与许可证 | 模型和数据 | 可验证训练权重 | 依赖与本环境 | 原始预算及单 GPU | 粒度与指标 |
|---|---|---|---|---|---|---|
| [Unstructured Pruning, ICLR 2024](https://github.com/xyshi2000/Unstructured-Pruning) | 公开，MIT | README 明确 CIFAR-10；[入口](https://raw.githubusercontent.com/xyshi2000/Unstructured-Pruning/main/main.py) 默认 Cifar10Net，也含 SEW-ResNet / CIFAR10-DVS 代码 | UNKNOWN；README 未给可验证预训练权重链接 | torch、torchvision、tensorboard、thop、spikingjelly 0.0.0.0.12；当前 torch 2.4.1+cu118，老版依赖兼容 UNKNOWN | 入口默认 search 800、finetune 200 epochs；是否完全等于论文所有设置 UNKNOWN；非分布式代码路径存在，单 GPU 实测 UNKNOWN | 单连接权重及神经元掩码；准确率、权重/神经元稀疏度及 SOP 等，细节需逐配置复核 |
| [SCA, ICML 2024](https://proceedings.mlr.press/v235/li24bz.html) | 论文公开；作者代码、许可证 UNKNOWN | [论文](https://raw.githubusercontent.com/mlresearch/v235/main/assets/li24bz/li24bz.pdf) 使用 VGG、Pre-ResNet；CIFAR-10/100、DVS-CIFAR10 等 | UNKNOWN | PyTorch/SpikingJelly，确切版本及当前环境兼容性 UNKNOWN | 论文称所有实验 300 epochs；单 GPU 要求与实际显存 UNKNOWN | 以脉冲活动进行卷积核级动态剪除与再生，训练后删除通道；报告准确率、参数、SynOps |
| [UniSparse, ICML 2026](https://github.com/xyshi2000/UniSparse) | 公开，MIT | README 含 CIFAR-10 的 CifarNet/Res19/Res18、DVS-CIFAR10 与 ImageNet Res18 | UNKNOWN；README 未给来源可核验的预训练权重 | spikingjelly 0.0.0.0.14、cupy-cuda12x、timm、torch；README 提到 numpy 兼容补丁；当前 CUDA build 11.8 与所列 CuPy CUDA12x 兼容 UNKNOWN | 原论文训练轮数 UNKNOWN；CIFAR 示例直接运行 python，单 GPU 实测 UNKNOWN；ImageNet 示例明确 4 GPU | 联合权重剪枝和脉冲稀疏化；准确率、权重/活动稀疏度与 SOP/能耗代理口径需按论文及配置核实 |

SCA 已证明活动引导的结构化通道剪枝属于已有研究；本轮无原创性宣称。A/B 校准样本属于训练来源，不能作为外部分布泛化验证。

## ROUND-014 补充
- [C-DPSS](https://github.com/JunLi0514/C-DPSS)，固定提交 `ea3a3b3dbef8746e030292fe03a276df18bbef22` 已克隆静态读取，包含 models、sparselearning、utils、main、datasets；未运行，不能确认全部路径完整可运行。树中无 LICENSE，许可 UNKNOWN。README 给 VGG16SNN/CIFAR100、300 epochs、channel/filter 动态稀疏、global magnitude death 和 global gradients growth 的示例；requirements 为 torch 2.5.1+cu121 / torchvision 0.20.1+cu121，现环境为 cu118，兼容 UNKNOWN。论文 DOI https://doi.org/10.1016/j.neucom.2026.134101 未成功打开，论文特定完整训练配置和权重 UNKNOWN。
- [SBC 预印本](https://arxiv.org/abs/2506.03996) 的当前 arXiv v3 标题为 *Spiking Brain Compression: Post-Training Second-order Compression for Spiking Neural Networks*；作者文本提到补充文件 `osbc_prune.py` 等，但本轮未核实公开仓库及其完整可用性，标记 UNKNOWN。二阶剪枝属于已有研究，不重命名为新创新。
