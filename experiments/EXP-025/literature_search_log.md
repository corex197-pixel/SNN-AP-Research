# 文献检索与访问记录
检索：`spiking neural network structured pruning membrane potential`、`spiking channel activity pruning threshold margin`、`SNN channel pruning sensitivity membrane potential`、`structured dynamic sparse training SNN channel regrowth`、`spiking neural network pruning temporal redundancy`、`"spiking activity based pruning" github`。关键词搜索是发现途径，以下仅援引可追溯论文/官方源码。

1. Yaxin Li, Qi Xu, Jiangrong Shen, Hongming Xu, Long Chen, Gang Pan, “Towards Efficient Deep Spiking Neural Networks Construction with Spiking Activity based Pruning”, ICML 2024, [PMLR 235, 29063–29073](https://proceedings.mlr.press/v235/li24bz.html), [全文](https://arxiv.org/html/2406.01072)。膜电位 L1 + 全局结构动态学习。
2. Jun Li 等, “C-DPSS: Channel dual-phase sparsity pruning framework for spiking neural networks”, Neurocomputing 696 (2026), [DOI 10.1016/j.neucom.2026.134101](https://doi.org/10.1016/j.neucom.2026.134101)；[出版商](https://www.sciencedirect.com/science/article/pii/S0925231226014992) 访问受限，ABSTRACT_ONLY；[源码](https://github.com/JunLi0514/C-DPSS/tree/ea3a3b3dbef8746e030292fe03a276df18bbef22) 静态可读。活动与权重的动态结构稀疏具有直接重复风险。
3. Shuiping Gou 等, “Dynamic spatio-temporal pruning for efficient spiking neural networks”, Frontiers in Neuroscience (2025), [DOI 10.3389/fnins.2025.1545583](https://doi.org/10.3389/fnins.2025.1545583)。已研究 DVS 中的动态时间冗余与权重稀疏，单纯引入时间信息不能视为新颖。
4. Yao 等, “Inherent Redundancy in Spiking Neural Networks”, ICCV 2023, [CVF全文](https://openaccess.thecvf.com/content/ICCV2023/html/Yao_Inherent_Redundancy_in_Spiking_Neural_Networks_ICCV_2023_paper.html)。阈值相关冗余已有研究，阈值距离不能直接宣称首创。

全文不可达与公开实现未定位均记 UNKNOWN；摘要不足以排除公式级重复。
