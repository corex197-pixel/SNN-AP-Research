# Codex科研执行任务：ROUND-023

# SNN结构化剪枝Baseline准入与科研机会审计

## 1. 任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-023
- 实验编号：EXP-025
- 科研决策：DEC-023，STOP + REASSESS
- 理论版本：v0.2.9
- 科研阶段：研究方向重新评估、候选Baseline静态准入
- 项目仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-022-exact-spike-contribution
- 本轮建议分支：research/round-023-structured-baseline-audit
- 上轮实际执行代码SHA：c494b5876de2dc1e0beb62c32137a0be63f4d19a
- 上轮选择锁SHA：ec26e79291b67f44d0fabdd2e3127963d79b9a1d
- 上轮最终结果SHA：7d5bdba3532ccb5b7dc3a8d69053c39314fe4a93
- 依赖实验：EXP-014、EXP-018、EXP-019、EXP-020、EXP-021、EXP-022、EXP-023、EXP-024

优先级：

P0：完成EXP-024科学状态归档。

P0：SCA与C-DPSS候选Baseline静态审查。

P1：确定可复现性、代码完整性、训练资源及潜在研究机会。

P2：审查Dynamic Spatio-Temporal Pruning作为备选。

本轮不训练、不微调、不运行GPU、不实现新的剪枝方法。

不能擅自决定正式切换Baseline。

## 2. ROUND-022最终科研判定

EXP-024保持COMPLETED。

所有准入检查及真实结构删除通过。

整体性能推进门槛失败。

本轮统一关闭cuDNN TF32及matmul TF32。

历史默认后端B0与本轮严格FP32 B0存在45张预测差异，不能混用精度口径。

RUN-001原重构误差超限及失败记录必须保留。

固定25%隐层神经元剪枝：

原模型参数量36720484。

真实紧凑模型参数量28280164。

实际删除8440320个参数。

结构压缩率22.98531795%。

第300轮固定U2父模型验证结果：

B0：

Accuracy=94.22222%。

CE=0.17273286。

SOP=86009551.08977778。

S0随机：

Accuracy=94.11111%。

CE=0.17192459。

SOP=82163586.0128889。

S1有效L1：

Accuracy=94.33333%。

CE=0.17019318。

SOP=82181961.79888889。

S2发放率：

Accuracy=93.73333%。

CE=0.17598948。

SOP=82187301.31133333。

S3静态单神经元损失：

Accuracy=87.66667%。

CE=0.38769187。

SOP=82157550.97866666。

M5条件边际损失：

Accuracy=93.97778%。

CE=0.17572403。

SOP=82170465.47622222。

M5-S1准确率差：

-0.35556个百分点。

95% CI [-0.75611,0.04444]。

M5-S2：

+0.24444个百分点。

95% CI [-0.24500,0.71111]。

M5-S3：

+6.31111个百分点。

95% CI [5.46667,7.11111]。

联合损失非加性：

S3单项Delta CE之和=-0.01830631。

真实联合Delta CE=+0.08407867。

差值=+0.10238498。

M5单项Delta CE之和=-0.00915424。

真实联合Delta CE=-0.00440404。

差值=+0.00475020。

H-024更新为SUPPORTED，限定指定网络线性读出及固定模型结构删除条件。

H-025更新为PARTIALLY_SUPPORTED：

条件边际损失重估相对静态单项损失排序的优势得到支持，但相对L1和FR的预注册性能门槛未通过。

保持原EXP-024整体门槛失败。

停止当前M5正式推进。

不开展额外剪枝率搜索、block size搜索或微调。

保留R101—R108精确读出及组合损失分析。

## 3. 保留既有科研资产

不得删除、覆盖或改变：

- AT-LIF历史Baseline及负面结果。
- UniSparse受控300轮训练模型及记录。
- EXP-020/021的MP-EFR实验。
- EXP-022的TP-EFR实验。
- EXP-023的SEAWP实验及训练准入偏差。
- EXP-024的结构化压缩模型、失败记录和精确损失分析。

UniSparse仍为当前已建立的Controlled Baseline。

新的候选Baseline在获得理论研究员审查及用户批准前，不得替换它。

## 4. 优先候选A：SCA

正式论文：

Towards Efficient Deep Spiking Neural Networks Construction with Spiking Activity based Pruning。

Yaxin Li, Qi Xu, Jiangrong Shen, Hongming Xu, Long Chen, Gang Pan。

ICML 2024。

PMLR 235，29063—29073。

论文主页：

https://proceedings.mlr.press/v235/li24bz.html

论文全文：

https://arxiv.org/pdf/2406.01072

优先核查：

1. 作者主页及正式论文中的代码链接。
2. OpenReview附件及补充材料。
3. 是否存在明确作者身份的官方实现。
4. 是否存在可追溯预训练权重。
5. 软件许可证及代码使用条件。
6. 实际训练入口、模型定义、数据预处理及配置。
7. 若只有第三方实现，明确标记第三方来源。
8. 不得将同名SCA的ANN注意力剪枝项目误认为本论文实现。

本轮理论研究员未确认作者官方代码，因此不得虚构仓库地址。

若找不到，记录检索位置、关键词及访问状态。

无法访问不等于不存在。

## 5. SCA核心理论审查

从原论文第4—5页：

Section 4.1。

Algorithm 1。

Section 4.2。

Equation (4)。

Section 4.3。

Section 4.4。

Section 5.1—5.4及相关图表。

抽取并核查：

SCA原通道重要性：

r[l,k] = sum_{n,t}(L1_norm(H[n,l,k,t])) / (N*T)

其中H为膜电位特征图。

不能将该指标错误描述为平均脉冲发放率。

特别核查：

- H表示充电后发放前膜电位，还是重置后的膜电位。
- 是否真的使用空间L1范数，而非空间平均。
- 全局通道排序是否存在空间尺度差异。
- 不同层的阈值和膜电位尺度是否相同。
- L1正则化具体作用对象。
- 初始剪枝比例p。
- 每次更新比例q。
- 更新间隔及调度。
- BN gamma梯度再生机制。
- 被掩码通道如何保持再生所需梯度。
- 残差连接的依赖处理。
- 训练后真实结构压缩方式。

论文没有明确给出的内容标记UNKNOWN。

不得未经证据确定实现细节。

若没有作者代码，单独区分论文明确公式与未来自行实现时的待定工程决策。

## 6. 优先候选B：C-DPSS

正式论文：

C-DPSS: Channel dual-phase sparsity pruning framework for spiking neural networks。

Neurocomputing，2026。

出版商页面：

https://www.sciencedirect.com/science/article/pii/S0925231226014992

公开代码：

https://github.com/JunLi0514/C-DPSS

理论研究员已核验的候选代码SHA：

ea3a3b3dbef8746e030292fe03a276df18bbef22

本轮优先使用该固定版本进行静态审查。

不得自动使用后续未固定版本。

仓库已确认存在：

main.py

args.py

datasets.py

models/vggsnnTbn.py

models/Monitor.py

sparselearning/core_dst_chase.py

sparselearning/flops.py

sparselearning/funcs.py

sparselearning/norm_funcs.py

requirements.txt

README.md

README代表性命令使用CIFAR100、vgg16snn、300 epochs、density=0.3，并启用多项稀疏训练及通道剪枝选项。

不得将README配置直接视为已验证的论文代表性工作点。

逐项检查：

- 参数解析及默认值。
- 初始稀疏结构生成。
- 权重非结构化剪枝与再生。
- 通道级剪枝阶段。
- 通道重要性计算。
- 结构稀疏度调度。
- 层级预算分配。
- mask更新顺序。
- 神经元模型和替代梯度。
- 是否实际缩小模型结构。
- 稀疏训练过程中是否仍使用稠密算子。
- SOP及FLOPs定义。
- train/validation/test数据协议。
- 是否利用正式测试集选择结构、阈值或最佳模型。
- 环境依赖及与现有RTX4090/CUDA环境兼容性。

README包含`--indicate_method test`。

必须追踪该参数真实代码语义，不能仅凭名称认定存在测试集泄漏。

C-DPSS仓库根目录目前未确认明确许可证。

在许可证及作者授权未明确前：

仅进行合法的源码阅读及静态审查。

不得将其源码复制提交至本项目仓库。

不得假定公开GitHub仓库自动允许复制、修改及再分发。

## 7. 备选候选C：Dynamic Spatio-Temporal Pruning

论文：

Dynamic spatio-temporal pruning for efficient spiking neural networks。

Shuiping Gou等。

Frontiers in Neuroscience，2025。

DOI：

10.3389/fnins.2025.1545583。

论文：

https://doi.org/10.3389/fnins.2025.1545583

代码：

https://github.com/gzxdu/SNN_Pruning

理论研究员已核验的代码SHA：

a137f1bca476d49b2c3d403c6d37688548d81a49

重点审查：

train_spiking_snn_cifar10dvs_VGGSNN.py

train_spiking_snn_cifar10dvs_VGGSNN_T.py

config_lth.py

utils_for_snn_lth_lamp_2stage.py

utils_for_snn_lth_lamp_VGGSNN.py

确认实际空间剪枝粒度、动态时间剪枝定义、数据集需求及依赖。

检查是否必须额外下载CIFAR10-DVS。

本轮不下载大型数据集。

不启动训练。

只作为备选进行轻量静态审查。

## 8. 科研机会审查

本轮不确定最终新算法。

以下科学问题仅作为审查对象：

Q-013A：

膜电位绝对幅度与通道实际任务贡献是否存在系统性失配？

对应SCA Eq.(4)。

Q-013B：

不同空间分辨率的通道膜电位L1范数进行全局比较时，是否可能产生尺度偏差？

必须首先确认原文及实际代码是否进行了空间归一化。

Q-013C：

动态剪枝和再生阶段能否充分利用SNN时间依赖信息？

重点核查已有方法是否已经考虑这一问题。

Q-013D：

结构化通道预算相同的情况下，是否存在不同实际参数压缩量及SOP节省？

重点考虑真实前驱与后继层的结构依赖。

对每个问题输出：

- 对应论文公式或源码位置。
- 原方法已采取的缓解机制。
- 潜在数学局限。
- 可观测变量。
- 最低成本验证方式。
- 可能的竞争性解释。
- 已有研究重复风险。
- 是否值得后续继续研究。

本轮不得将尚未验证的理论问题标记为实际缺陷。

## 9. R109—R112理论记录

追加理论版本v0.2.9。

R109：

对于交叉熵关于logits的Hessian：

H = diag(p)-p*p^T

其中p=softmax(z)。

R110—R111：

神经元联合删除时二阶损失包含不同贡献向量之间的交叉项。

该性质并非SNN独有。

EXP-024已经验证静态单项损失和实际联合损失明显不同。

R112：

对于硬阈值发放s=Theta(H-Vth)，若：

abs(delta_H)<=epsilon

且：

abs(H-Vth)>epsilon

则原脉冲状态保持不变。

该结论只适用于指定单时间步、给定发放前膜电位与扰动条件。

不能据此断言高发放敏感神经元一定重要。

不实现新的阈值距离剪枝算法。

## 10. 文献检索与重复风险审查

如果环境具备联网能力，检索：

"spiking neural network structured pruning membrane potential"

"spiking channel activity pruning threshold margin"

"SNN channel pruning sensitivity membrane potential"

"structured dynamic sparse training SNN channel regrowth"

"spiking neural network pruning temporal redundancy"

优先核查2023—2026年研究。

必须记录：

论文标题、作者、正式发表年份、来源、DOI或可信链接。

摘要可用于初筛，但不得据此宣称核心公式完全不同。

如果全文无法访问，明确标注ABSTRACT_ONLY或BLOCKED。

特别核查C-DPSS及其他基于活动的结构化动态稀疏研究，避免重复设计已有机制。

不得编造文献。

## 11. Baseline准入审查表

对每个候选填写：

- 论文与代码是否能够对应。
- 官方代码身份。
- 固定Commit SHA。
- 代码许可证状态。
- 训练入口完整性。
- 模型及神经元实现。
- 数据集可获得性。
- 预训练权重来源。
- 权重SHA256（仅实际取得时）。
- 完整训练所需Epoch。
- 时间步T。
- Batch Size。
- 参数量。
- 软件依赖。
- CUDA后端需求。
- 是否必须修改现有环境。
- 正式测试集隔离可行性。
- 真实结构化压缩支持。
- 公平对照实验可实现性。
- 代表性实验复现方案。
- 潜在训练资源风险。
- 创新性重复风险。

分类采用：

VERIFIED。

PARTIALLY_VERIFIED。

UNKNOWN。

BLOCKED。

静态准入通过不代表正式训练或论文结果已复现。

不得虚构显存、吞吐量、运行时间或训练准确率。

## 12. 候选评分规则

每项0—10分。

权重：

- 科学问题研究空间：20%。
- 学术创新潜力：20%。
- SNN特异性：10%。
- 代码与权重可获得性：15%。
- 实验可复现性：15%。
- 计算资源适配性：10%。
- 公平对照及真实结构压缩可实现性：10%。

评分必须说明证据及不确定性。

未知信息不得直接按已验证事实评分。

代码许可证不明确、核心实现无法获得或训练预算无法承担，应单独列为准入阻塞项。

评分不是自动切换Baseline的依据。

## 13. 资源和操作限制

本轮仅CPU静态审查。

CPU累计计算上限45分钟。

GPU使用上限0秒。

不得：

- 启动CUDA模型或GPU测试。
- 开展训练、微调及正式推理实验。
- 安装大型依赖。
- 下载大型数据集或权重。
- 修改系统驱动。
- 修改现有UniSparse或AT-LIF实验环境。
- 未经许可证审查导入第三方源码。
- 读取正式测试集。
- 实现额外创新算法。

可以在临时位置进行必要的合法只读源码审查。

记录实际访问失败与资源消耗。

如果无法联网，保留已知固定来源并报告阻塞。

不得用虚构文件填补缺失代码。

## 14. 下一阶段建议

针对最值得推荐的候选，提出一份具体但尚未执行的最小工程准入方案。

内容包括：

- 最小模型结构。
- 推荐数据集及数据隔离。
- 原论文代表性配置。
- 关键机制正确性测试。
- CPU冒烟测试。
- 单GPU前向和反向验证。
- 后续资源基准测量。
- 是否可以利用来源可靠的预训练权重。
- 预计需要用户批准的实验范围。

不直接启动后续阶段。

如果所有候选均不满足准入要求，明确建议保留UniSparse或暂缓科研方向切换，不得强行选出新Baseline。

## 15. 科研状态更新

追加：

research/ROUND-023-TASK.md

research/STRUCTURED_BASELINE_COMPARISON.md

更新：

research/THEORY.md

research/HYPOTHESES.md

research/RESEARCH_STATE.md

research/EXPERIMENT_PLAN.md

research/EXPERIMENT_REGISTRY.csv

research/DECISION_LOG.md

research/LITERATURE.md

保留EXP-024原性能门槛失败。

ESC-JP当前M5路线标记STOPPED。

UniSparse保留为当前Controlled Baseline。

候选Baseline标记UNDER_REVIEW。

不能擅自升级为ENGINEERING_ADMITTED。

## 16. 结果交付

建立独立目录：

experiments/EXP-025/

至少保存：

- TASK.md
- config.json
- environment.txt
- code_version.json
- commands.sh
- status.json
- candidate_sources.json
- baseline_comparison.csv
- license_audit.md
- SCA_formula_audit.md
- SCA_code_availability.md
- CDPSS_code_audit.md
- CDPSS_formula_mapping.md
- dynamic_st_pruning_audit.md
- data_protocol_audit.md
- research_opportunity_matrix.csv
- literature_search_log.md
- admission_decision.json
- next_stage_proposal.md
- resource_usage.csv
- REPORT.md
- limitations.md
- run.log

所有源码位置采用固定SHA对应的文件路径及行号。

未确认信息明确标记UNKNOWN。

不得创建虚构训练日志、实验准确率或资源测量结果。

## 17. GitHub交付

从ROUND-022最终提交创建独立分支。

读取并遵守AGENTS.md。

开始审查前提交任务及固定协议。

结束后提交全部真实审查结果、科研状态及决策建议。

不修改main。

不force push。

不覆盖历史实验。

最终返回：

- GitHub实验分支链接。
- 审查代码及任务提交SHA。
- 最终结果提交SHA。
- EXP-025执行状态。
- SCA官方代码与权重是否确认。
- C-DPSS代码、许可证及核心机制审查结果。
- 三个候选Baseline评分和准入状态。
- 最值得进一步研究的1—2个科学问题。
- 推荐Baseline及其依据。
- 主要复现风险与资源阻塞。
- 后续最小工程准入计划。
- 报告路径。
- 实际CPU/GPU资源消耗。

本轮是研究方向重新评估，不是再次试错新算法。

获得用户授权后执行。