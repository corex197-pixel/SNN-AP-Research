# 文献：初步风险排查，未完成系统查重

- Bu, Shi, Yu (NeurIPS 2025), [Activity Pruning for Efficient Spiking Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2025/hash/fa12d67b5939c37ea8a4659c31a08d2c-Abstract-Conference.html).
- Han, Srinivasan, Roy (CVPR 2020), [RMP-SNN](https://arxiv.org/abs/2003.01811).
- Hao, Bu, Ding, Huang, Yu (AAAI 2023), [Reducing ANN-SNN Conversion Error through Residual Membrane Potential](https://arxiv.org/abs/2302.02091).
- Shi, Ding, Hao, Yu (ICLR 2024), [Towards Energy Efficient Spiking Neural Networks](https://proceedings.iclr.cc/paper_files/paper/2024/hash/dd3c889922df2112a5b1769e3c19e28e-Abstract-Conference.html).

残余膜电位、软重置及相关误差已有文献；不宣称原创性。


ROUND-007 初步创新性风险排查，非系统查重：

- Wang et al., [Adaptive Calibration: A Unified Conversion Framework of Spiking Neural Network](https://arxiv.org/abs/2412.16219).
- Shi, Bu, Yu, [UniSparse: Combining Weight Pruning and Spike Sparsification in Spiking Neural Networks](https://proceedings.mlr.press/v306/shi26a.html), ICML 2026.
- [Optimizing the Energy Consumption of Spiking Neural Networks for Neuromorphic Applications](https://doi.org/10.3389/fnins.2020.00662), 2020.

## ROUND-009 相关工作与创新风险
- Thomas Louis et al., On Reducing Activity with Distillation and Regularization for Energy Efficient Spiking Neural Networks (2024), https://arxiv.org/abs/2406.18350
- Optimizing the Energy Consumption of Spiking Neural Networks for Neuromorphic Applications (2020), https://doi.org/10.3389/fnins.2020.00662
- Xinyu Shi et al., UniSparse: Combining Weight Pruning and Spike Sparsification in Spiking Neural Networks, ICML 2026, https://proceedings.mlr.press/v306/shi26a.html
知识蒸馏、活动正则化和SynOps优化已有研究；本方案是借鉴成熟思想的AT-LIF先导改进，独立学术创新性尚未证明。

## ROUND-010 BN相关工作与创新风险
- EagleEye: Fast Sub-Net Evaluation for Efficient Neural Network Pruning, ECCV 2020, https://arxiv.org/abs/2007.02491
- Wu and Johnson, Rethinking "Batch" in BatchNorm, 2021, https://arxiv.org/abs/2105.07576
- Membrane Potential Batch Normalization for Spiking Neural Networks, ICCV 2023, https://openaccess.thecvf.com/content/ICCV2023/html/Guo_Membrane_Potential_Batch_Normalization_for_Spiking_Neural_Networks_ICCV_2023_paper.html.
- A Free Lunch From ANN: Towards Efficient, Accurate Spiking Neural Networks Calibration, ICML 2021, https://proceedings.mlr.press/v139/li21d.html
普通BN统计重新校准已有相关研究，逐层候选方法尚未证明独立创新性。

## ROUND-011 相关工作和创新风险
- Learnable Surrogate Gradient for Direct Training Spiking Neural Networks, IJCAI 2023, https://ijcai.org/proceedings/2023/335
- SpQuant-SNN: ultra-low precision membrane potential with sparse activations unlock the potential of on-device spiking neural networks, 2024, https://doi.org/10.3389/fnins.2024.1440000
- Kook et al., Stabilizing Direct Training of Spiking Neural Networks: Membrane Potential Initialization and Threshold-robust Surrogate Gradient, 2025, https://arxiv.org/abs/2511.08708
- ELVIA analysis, https://elvia-atlas.org/atlas/cases/neurips2025-0028-activity-pruning-for-efficient-spiking-neural-networks.html ，为待独立审查的外部分析，不是同行评审论文；2026-11-27尝试读取时外部工具返回cache miss，未能独立核验其内容。可学习阈值、替代梯度及梯度路径已有研究；本轮尚未证明独立学术创新性。

## ROUND-012 梯度尺度相关工作与创新风险
- Esser et al., Learned Step Size Quantization, ICLR 2020, https://arxiv.org/abs/1902.08153
- Zenke and Vogels, The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks, 2021, https://pmc.ncbi.nlm.nih.gov/articles/PMC7619566/
- Learnable Surrogate Gradient for Direct Training Spiking Neural Networks, IJCAI 2023, https://ijcai.org/proceedings/2023/335
- Kook et al., Stabilizing Direct Training of Spiking Neural Networks: Membrane Potential Initialization and Threshold-robust Surrogate Gradient, https://arxiv.org/abs/2511.08708
共享参数梯度缩放与替代梯度尺度已有研究，mean改sum本身不宣称独立创新。

## ROUND-013 existing work
SCA is prior structured spike-activity channel pruning and regeneration during training: https://proceedings.mlr.press/v235/li24bz.html . UniSparse combines weight pruning and spike sparsification: https://github.com/xyshi2000/UniSparse . Direct task-loss sensitivity and structural pruning are established ideas; EXP-014 does not claim invention. See BASELINE_CANDIDATES.md for verification and unknowns.

## ROUND-014
UniSparse [ICML 2026](https://proceedings.mlr.press/v306/shi26a.html) 的联合连接与脉冲稀疏是既有方法。C-DPSS [代码](https://github.com/JunLi0514/C-DPSS)包含动态稀疏结构；[SBC](https://arxiv.org/abs/2506.03996)为已有二阶后训练压缩研究。原始公式映射见 UNISPARSE_BASELINE.md；不得将这些既有思想单纯更名为创新。

## ROUND-015 相关工作与近似审查
[UniSparse ICML 2026](https://proceedings.mlr.press/v306/shi26a.html) 第7页明确使用初始结构常数近似动态连接/活动映射，以缓解正反馈；本轮仅验证这种既有近似与有效扇出在合成情形下的差别。[Optimizing the Energy Consumption of Spiking Neural Networks for Neuromorphic Applications](https://doi.org/10.3389/fnins.2020.00662) 和 [Backpropagation With Sparsity Regularization for Spiking Neural Network Learning](https://doi.org/10.3389/fnins.2022.760298) 体现既有 SNN 能耗及稀疏正则研究；[Unstructured Pruning Framework ICLR 2024](https://github.com/xyshi2000/Unstructured-Pruning) 已研究联合稀疏。不可将有效连接加权或联合稀疏直接宣称为首创。

ROUND-016 retains the author’s explicit initial-structure approximation; the pilot tests original UniSparse and does not establish novelty for effective fanout. Official CifarNet nonzero λ remains UNKNOWN after limited source audit.

ROUND-017 R69–R73 is symmetric two-order cost attribution (Shapley-style accounting), not a new pruning algorithm and not a causal estimate of training interventions; UniSparse's initial-structure approximation remains the author's explicit design.


## ROUND-018 references and novelty boundary
- https://doi.org/10.3389/fnins.2020.00662 — Optimizing the Energy Consumption of Spiking Neural Networks for Neuromorphic Applications.
- https://doi.org/10.3389/fnins.2022.760298 — Backpropagation With Sparsity Regularization for Spiking Neural Network Learning.
- https://ieeexplore.ieee.org/document/11132570/ — IEEE record; automated bibliographic inspection blocked by site verification.
- https://proceedings.mlr.press/v306/shi26a.html — UniSparse (Shi, Bu, Yu; ICML 2026).
Related SynOps/activity/weight-sparsity ideas exist. No first-discovery claim for normalized fanout weighting or convex combination.
