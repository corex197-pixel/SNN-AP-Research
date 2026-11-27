# ROUND-013 候选 Baseline 静态可复现性审查

核查方式：2026-11-27 阅读公开代码仓库、README、代码入口及论文；未安装依赖、下载大数据或运行训练。UNKNOWN 表示未在所核查的来源中确认，而非断言不存在。

| 项目 | 代码与许可证 | 模型和数据 | 可验证训练权重 | 依赖与本环境 | 原始预算及单 GPU | 粒度与指标 |
|---|---|---|---|---|---|---|
| [Unstructured Pruning, ICLR 2024](https://github.com/xyshi2000/Unstructured-Pruning) | 公开，MIT | README 明确 CIFAR-10；[入口](https://raw.githubusercontent.com/xyshi2000/Unstructured-Pruning/main/main.py) 默认 Cifar10Net，也含 SEW-ResNet / CIFAR10-DVS 代码 | UNKNOWN；README 未给可验证预训练权重链接 | torch、torchvision、tensorboard、thop、spikingjelly 0.0.0.0.12；当前 torch 2.4.1+cu118，老版依赖兼容 UNKNOWN | 入口默认 search 800、finetune 200 epochs；是否完全等于论文所有设置 UNKNOWN；非分布式代码路径存在，单 GPU 实测 UNKNOWN | 单连接权重及神经元掩码；准确率、权重/神经元稀疏度及 SOP 等，细节需逐配置复核 |
| [SCA, ICML 2024](https://proceedings.mlr.press/v235/li24bz.html) | 论文公开；作者代码、许可证 UNKNOWN | [论文](https://raw.githubusercontent.com/mlresearch/v235/main/assets/li24bz/li24bz.pdf) 使用 VGG、Pre-ResNet；CIFAR-10/100、DVS-CIFAR10 等 | UNKNOWN | PyTorch/SpikingJelly，确切版本及当前环境兼容性 UNKNOWN | 论文称所有实验 300 epochs；单 GPU 要求与实际显存 UNKNOWN | 以脉冲活动进行卷积核级动态剪除与再生，训练后删除通道；报告准确率、参数、SynOps |
| [UniSparse, ICML 2026](https://github.com/xyshi2000/UniSparse) | 公开，MIT | README 含 CIFAR-10 的 CifarNet/Res19/Res18、DVS-CIFAR10 与 ImageNet Res18 | UNKNOWN；README 未给来源可核验的预训练权重 | spikingjelly 0.0.0.0.14、cupy-cuda12x、timm、torch；README 提到 numpy 兼容补丁；当前 CUDA build 11.8 与所列 CuPy CUDA12x 兼容 UNKNOWN | 原论文训练轮数 UNKNOWN；CIFAR 示例直接运行 python，单 GPU 实测 UNKNOWN；ImageNet 示例明确 4 GPU | 联合权重剪枝和脉冲稀疏化；准确率、权重/活动稀疏度与 SOP/能耗代理口径需按论文及配置核实 |

SCA 已证明活动引导的结构化通道剪枝属于已有研究；本轮无原创性宣称。A/B 校准样本属于训练来源，不能作为外部分布泛化验证。
