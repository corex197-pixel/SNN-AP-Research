# Codex 科研执行任务：ROUND-014

# 旧研究路线归档与 UniSparse 新 Baseline 工程准入审查

## 1. 任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-014
- 科研阶段：新Baseline阶段0，理论与工程准入审查
- 科研决策：DEC-014，STOP + PIVOT
- 历史理论版本：v0.1.9
- 新理论记录版本：v0.2.0
- 新科学问题：Q-010
- 实验编号：EXP-015
- 项目仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-013-structured-channel-pruning
- 本轮建议分支：research/round-014-baseline-transition
- 上轮执行SHA：45cc4bd67e8d71329be1d235a54f1942dd2731f1
- 上轮结果SHA：55c595d38cfa75e27977b04828471315da91d306
- 上轮最新远程SHA：bb840b54f7d27578049365a9650f16fb12a5d52b
- 原始AT-LIF导入SHA：52878cff06f6378fa155c032b7acd23883a72e4a

优先级：

P0：历史研究状态归档与新Baseline版本核查。

P1：论文—代码对应、环境检查和稀疏计算正确性测试。

P2：受控短程GPU资源测量。

本轮不实现新的创新算法，不开展完整模型训练，不使用官方测试集。

## 2. ROUND-013正式科研判定

EXP-014为COMPLETED。

原始结果：

- B0：Accuracy=91.7111%，Parameters=1084605，SOPs=55023962.133。
- P-Random：Accuracy=85.5333%，Parameters=987725，SOPs=53002756.459。
- P-L1：Accuracy=87.2222%，Parameters=987725，SOPs=56979476.860。
- P-FR：Accuracy=82.7333%，Parameters=987725，SOPs=51849096.032。
- P-Loss：Accuracy=79.2222%，Parameters=987725，SOPs=54270553.909。

四组真实删除参数均为96880，占原模型约8.93%。

全部结构等价性检查通过。

H-014正式判定：

CONTRADICTED，仅限本次三个指定残差模块、内部通道删除25%、不微调及预设先导性能门槛。

不得扩大为SNN不存在结构冗余。

H-015正式判定：

CONTRADICTED，仅限本轮固定样本、单通道损失排序及同时删除协议。

P-Loss相对P-L1准确率下降8个百分点，配对95% CI=[-9.2,-6.755]个百分点。

P-Loss相对P-FR准确率下降3.5111个百分点，配对95% CI=[-4.7111,-2.4]个百分点。

两项均为未经多重比较校正的样本级区间。

保留H-012 SUPPORTED及H-013 INSUFFICIENT_EVIDENCE。

保留全部其他历史科研假设及适用边界。

## 3. 新增理论研究记录

R51：

对n个通道独立均匀选择两个大小均为k的集合，其重合率期望为k/n。

ROUND-013最低25%实际删除损失排序的集合重合率分别为0.25、0.25、0.21875。

不能据此证明两组排序严格等效于随机排序。

R52：

设D(S)=L(f_without_S)-L(f)。

定义双通道交互：

J_ij=D({i,j})-D({i})-D({j})。

一般情况下：

D(S)不等于sum_c_in_S D({c})。

现有数据尚未在相同样本上直接测量联合删除交互误差。

理论研究员根据原始通道数据独立计算得到：

P-Random：

A集单通道D之和=0.1542770955。

B集单通道D之和=-0.2230751020。

P-L1：

A集=0.0410373723。

B集=-0.2863024320。

P-FR：

A集=0.1351848697。

B集=-0.2919633032。

P-Loss：

A集=-0.3565105177。

B集=-0.3616315147。

这些数字为既有原始数据的独立复算。

A/B的单通道损失和不能与验证集联合损失直接相减并称为交互项。

R53：

C(M,s)=sum_{l,b,t,i,j}(M[l,i,j]*s[l,t,b,i])/B。

M表示有效连接的二值存在性，s表示实际输入脉冲活动。

该公式为指定事件驱动连接计数口径。

R54—R55：

Delta_C_structure=C(M_new,s_old)-C(M_old,s_old)。

Delta_C_activity=C(M_new,s_new)-C(M_new,s_old)。

Delta_C=Delta_C_structure+Delta_C_activity。

该分解是统一连接索引空间中的代数恒等式，不是经过验证的新剪枝方法。

ROUND-013观察到P-L1参数减少8.93%，但SOPs增加3.55%。

结构变化引起的活动重分布是合理解释之一，实际分解尚未执行。

R56：

对于卷积连接，可用有效连接二值卷积核对二值活动执行卷积求和，统计指定范围内的理论活动运算量。

需要正确处理stride、padding、dilation、groups及真实连接掩码。

若候选模型存在非二值、多级或有符号活动，须按真实硬件计算假设重新确定活动定义，不能直接套用x>0。

## 4. 旧研究路线归档

停止主动推进以下方案：

- AT-LIF累计输出幅值补偿。
- 静态及动态边际阈值搜索。
- 原预算约束蒸馏微调。
- 普通和逐层BN校准。
- ARTG代理梯度。
- ROUND-013固定25%内部通道删除方案。

停止推进不表示相关方法在所有条件下无效。

保留原AT-LIF代码、权重、日志、数据划分、科研记录及所有历史实验。

在research/BASELINE_TRANSITION.md记录：

1. 已完成的十三轮科研工作。
2. 得到支持的科研结论。
3. 获得反对证据的条件性结论。
4. 尚不能判断的假设。
5. 停止投入各方案的原因。
6. 现有可复用的科研资产。
7. 新Baseline选择原因。

不得删除、移动或覆盖历史实验及大型检查点。

不将尚未形成独立创新误写为已具备发表成果。

## 5. 新Baseline候选身份

优先候选：

UniSparse: Combining Weight Pruning and Spike Sparsification in Spiking Neural Networks。

作者：Xinyu Shi、Tong Bu、Zhaofei Yu。

ICML 2026。

正式论文：

https://proceedings.mlr.press/v306/shi26a.html

官方仓库：

https://github.com/xyshi2000/UniSparse

理论研究员实际核查的官方代码版本：

b5b3da4fcd2e7abcd2c20c93d41cf63ef3201daa

已确认MIT许可证。

已确认存在：

- main.py
- models/cifarnet.py
- models/resnet.py
- models/submodules/layers.py
- utils/utils.py
- configs/cifar10_cifarnet.yaml
- configs/cifar10_res18.yaml
- configs/cifar10_res19.yaml

其他文件及调用关系以真实仓库为准。

不得把UniSparse作为已经得到可靠复现的Baseline。

目前只是优先准入候选。

## 6. 新代码管理

不得破坏现有Activity-Pruning-SNN-main目录。

将UniSparse官方源代码以可追溯方式放置在独立目录或独立受控工作区。

记录：

- 官方仓库URL。
- 官方Commit SHA。
- 导入方式。
- 官方许可证。
- 本地代码修改记录。
- 原始版本和适配版本之间的Git Diff。

可以采用独立目录保留原始官方实现及外围controlled适配层。

具体目录依据仓库现状选择，不进行无关重构。

不擅自将新研究仓库合并到main。

## 7. 论文与实际代码对应审查

读取UniSparse正式论文及作者可获得的补充材料。

若无法获取全文，记录访问失败原因，不通过二手总结编造原公式。

在research/UNISPARSE_BASELINE.md记录：

1. 论文完整书目信息。
2. 核心研究问题。
3. 神经元模型及前向动力学。
4. 权重稀疏化机制。
5. 脉冲活动稀疏化机制。
6. 权重阈值定义及更新方式。
7. BN仿射参数的作用。
8. 局部惩罚如何计算。
9. 联合稀疏优化目标。
10. 替代梯度及detach规则。
11. 权重和活动稀疏化之间的依赖关系。
12. 训练、测试与最终导出流程。
13. SOPs、权重密度及FR计算方式。
14. 原论文实验设置和官方代码对应配置。

每条核心数学公式保留原始论文编号，并标注对应代码文件、函数和行号。

论文公式、代码实际行为和独立推导必须区分。

重点读取真实实现中的：

- SparseConv2d。
- SparseBN。
- local_forward。
- calc_penalty。
- penalize。
- clear_penalty。
- sparse_weight。
- sparse_mask。
- SOPMonitor。

若名称或实现与实际版本不同，以真实源码为准并记录。

不得在理论研究员完成深度分析前擅自认定这些机制存在缺陷。

## 8. 固定工程准入配置

优先评估官方CifarNet + CIFAR-10。

不直接使用大型ImageNet或DVS数据。

官方入口已确认的默认配置包括：

- epochs=300。
- T=4。
- batch_size=256。
- seed=12450。
- lr=0.1。
- optimizer=sgd。
- weight_decay=0。
- lmbda_weight=0。
- lmbda_neuron=0。

需要核实实际配置文件、命令行覆盖、论文表格和内部参数组。

本轮仅使用这些值作为工程准入配置来源，不意味着以上默认值就是论文所有正式实验设置。

优先使用已经存在的CIFAR-10训练数据。

不下载官方测试集或访问其内容。

可复用旧项目的45000/4500/500数据划分文件用于受控实验准备。

本轮不进行4500验证集的正式性能评价。

## 9. 环境兼容性审查

实际核查：

- Python版本。
- PyTorch版本及CUDA build。
- CUDA Driver及Toolkit。
- CuPy版本。
- SpikingJelly版本。
- timm版本。
- numpy版本。
- torchvision版本。
- 单GPU可用性。

UniSparse官方README要求：

spikingjelly==0.0.0.0.14

cupy-cuda12x

并提到np.int兼容问题。

当前旧项目环境不得被破坏。

禁止直接在已稳定运行旧项目的环境中升级或卸载核心依赖。

如需创建独立虚拟环境，优先使用现有本地依赖缓存，并记录完整版本。

涉及大型下载、驱动修改、系统级安装或明显超出预算的依赖操作时停止并报告。

不得默认CUDA Driver支持某个版本就等于PyTorch、CuPy、运行时和内核编译全部兼容。

如果需要将CuPy backend改为torch backend：

- 只能作为明确标注的工程兼容性对照。
- 不得默认为论文等价复现。
- 必须对照验证前向和梯度语义。
- 未经理论确认不得将其用于正式模型训练。

## 10. 数据与评估入口审查

检查官方训练过程中是否每轮使用官方测试集，以及是否据此保存最佳模型。

检查训练结束后是否自动执行测试集评价。

检查eval阶段是否仍更新稀疏惩罚统计、BN状态或其他自定义状态。

为未来正式复现准备独立受控入口：

- 使用已有训练数据划分。
- 从训练集独立划分验证集。
- 官方测试集保持隔离。
- 不依赖测试集选择模型或超参数。
- 验证阶段不污染训练更新状态。
- 记录原作者协议与Controlled协议的差异。

本轮只允许必要的入口准备及测试，不运行完整训练。

## 11. 权重与活动稀疏化正确性检查

基于实际UniSparse实现完成最小合成测试。

需要验证：

1. 所有可训练参数和状态buffer的实际含义。
2. 稀疏权重如何产生。
3. 有效连接掩码与实际前向权重是否一致。
4. 权重阈值为零时的前向行为。
5. 不同有效连接状态下的卷积输出。
6. 神经元脉冲活动的定义。
7. 局部惩罚状态如何累计和清理。
8. 权重惩罚是否实际改变对应状态。
9. 活动惩罚是否实际改变对应参数或buffer。
10. 执行eval是否发生未经预期的状态更新。
11. AMP跳步与手动稀疏更新之间的关系。
12. 单卡设备隔离是否正确。

不能仅凭某个loss为非零就宣称有效稀疏干预已经发生。

必须记录真实被修改的参数和buffer。

合成结果标记为UNIT_TEST或SMOKE_TEST，不作为正式任务性能结果。

## 12. SOPs统计口径专项审查

检查UniSparse的实际SOPMonitor是否使用真实sparse_mask或等价有效连接表示。

不能假定其与旧AT-LIF的全1卷积计数相同。

至少使用三个确定性小型卷积场景：

A：稠密权重、稀疏脉冲。

B：稀疏权重、相同脉冲。

C：稀疏权重、不同脉冲。

计算：

- 原作者SOPMonitor的结果。
- 独立有效连接逐元素计数结果。
- 两者绝对误差和相对误差。

若原统计口径排除某类运算，独立计数必须采用相同范围。

此外单独记录完整计算口径的差别，不静默替换原作者指标。

验证R54的结构和活动分解在人工小型实例上的代数正确性。

如果使用真实随机初始化网络进行验证，结果仅用于机制和指标正确性检查，不代表训练后实际剪枝收益。

明确区分：

- 总模型参数量。
- 非零有效权重数。
- 权重稀疏率。
- 脉冲发放率。
- 理论SOPs。
- 实际张量尺寸。
- GPU实测时间。

不能将非结构化零权重数量直接解释为普通GPU上的实际推理加速。

## 13. EXP-015 GPU资源测量

只有环境检查、前向检查和稀疏状态检查通过后才能执行。

使用一张符合实验室共享规定的RTX4090。

优先固定官方CifarNet。

本轮预设工程配置：

- CIFAR-10训练来源样本。
- T=4。
- seed=12450。
- batch_size=256，若不满足资源要求则报告BLOCKED，不自动改批量。
- 官方数据增强规则。
- 官方优化器及参数组。
- 官方AMP协议。

只进行必要的短程前向、反向及优化器更新测量。

最多10个预热Batch和30个测量Batch。

不得运行完整Epoch。

测量：

- 数据加载时间。
- 前向耗时。
- 反向耗时。
- 优化器更新耗时。
- 稀疏惩罚计算及手动更新耗时。
- 单批次总墙钟时间。
- 峰值allocated及reserved显存。
- 实际成功优化器步数。
- AMP跳步。
- 权重阈值及其他关键状态的有限性。

稀疏惩罚系数优先使用论文和官方配置中明确可核实的单个代表性配置。

不得自行进行lambda搜索。

如果尚未确认正式论文使用的非零lambda，允许完成零惩罚路径的资源测量，并将非零惩罚正式资源测量标记BLOCKED。

允许使用明确标记的非零合成测试系数检查惩罚代码是否有作用，但不能将其当作正式论文配置或正式训练性能。

依据实测Batch耗时，估算单Epoch和300Epoch的运行成本。

明确估算假设和未包含的验证、保存及共享负载开销。

随机初始化的短程速度不保证与完整训练各阶段完全一致。

本轮生成的短程模型不得直接作为正式Baseline起点。

## 14. 准入判定规则

满足下列条件时，UniSparse可进入下一轮完整复现资源审批：

1. 官方版本和许可证可追溯。
2. 论文及核心代码能够建立对应关系。
3. 现有数据能够正常使用。
4. 独立环境依赖兼容。
5. 单GPU前向与反向能够执行。
6. 权重与活动稀疏化机制通过必要的正确性测试。
7. 主要SOPs统计口径明确。
8. 不存在未经授权的测试集访问。
9. 已取得真实资源测量依据。
10. 已识别论文级创新性审查所需的原始公式和机制。

如果仅环境或工程准入部分通过，记录PARTIAL。

若无法满足关键条件，记录BLOCKED，不继续扩大投入。

准入通过不代表Baseline复现完成，也不代表新方法具有创新性。

## 15. 备选Baseline补充审查

仅补充静态信息，不实现其他模型或启动训练。

C-DPSS：

C-DPSS: Channel Dual-phase Sparsity Pruning Framework for Spiking Neural Networks。

Neurocomputing，2026。

DOI：

https://doi.org/10.1016/j.neucom.2026.134101

官方仓库：

https://github.com/JunLi0514/C-DPSS

已核查官方版本：

ea3a3b3dbef8746e030292fe03a276df18bbef22

补充核查：

- 实际完整代码。
- 许可证。
- 动态结构剪枝和再生机制。
- 官方模型和数据配置。
- 原训练预算。
- 是否支持当前实验环境。

同时保留此前ICLR 2024 Unstructured Pruning、ICML 2024 SCA的审查记录。

补充OSBC预印本：

https://arxiv.org/abs/2506.03996

记录其公开代码是否可确认。

不得把已有研究中的联合稀疏、活动重要性、权重重要性切换或二阶剪枝直接重新命名为本项目创新。

## 16. 资源与终止条件

本轮需要用户重大方向调整授权。

建议：

- 一张符合实验室共享管理规定的RTX4090。
- GPU总占用上限10分钟。
- CPU累计计算上限45分钟。
- 正式训练轮数0。
- 仅允许规定范围内的工程测试及短程资源测量。

禁止：

- 完整300轮训练。
- 额外AT-LIF微调。
- 训练超参数搜索。
- 访问官方测试集。
- 自动增加GPU预算。
- 未经批准修改系统驱动或旧实验环境。
- 删除旧实验。
- 修改main或force push。

如果用户授权范围内无法完成候选准入，不自动转而运行其他候选模型。

保存全部阻塞原因。

## 17. 科研状态更新

新增：

research/ROUND-014-TASK.md

research/BASELINE_TRANSITION.md

research/UNISPARSE_BASELINE.md

更新：

- THEORY.md：保留全部历史理论，新增v0.2.0及R51—R56。
- HYPOTHESES.md：更新H-014和H-015为限定条件下的CONTRADICTED。
- RESEARCH_STATE.md：记录旧路线停止及新Baseline准入状态。
- BASELINE_CANDIDATES.md：补充真实候选核查结果。
- EXPERIMENT_PLAN.md。
- EXPERIMENT_REGISTRY.csv。
- DECISION_LOG.md：追加DEC-014。
- LITERATURE.md：追加真实相关文献及创新性风险。

本轮不凭空创建新的已验证性能假设。

Q-010保持待论文深入分析状态。

## 18. 实验交付

在experiments/EXP-015下保存：

- TASK.md
- config.json或config.yaml
- environment.txt
- upstream_version.json
- code_version.json
- commands.sh
- status.json
- dependency_audit.md
- paper_code_mapping.md
- dataset_protocol_audit.md
- sparse_mechanism_tests.json
- sop_count_tests.json
- state_isolation_tests.json
- resource_benchmark.csv
- resource_estimate.json
- baseline_admission.json
- run.log
- REPORT.md
- limitations.md

若某项未执行，明确标记NA及原因。

每次失败或重试使用唯一Run ID。

测试结果和真实正式实验结果必须分别记录。

## 19. GitHub交付及最终返回

正式运行前提交任务、固定配置及适配代码。

完成后提交全部真实记录。

返回：

1. GitHub实验分支链接。
2. UniSparse实际导入官方SHA。
3. 本轮执行代码SHA。
4. 最终结果提交SHA。
5. EXP-015执行状态。
6. 新环境的实际版本。
7. 论文核心公式与代码映射文件路径。
8. 权重及活动稀疏化检查结果。
9. SOPs统计一致性结果。
10. GPU资源测量结果。
11. 完整300轮训练的成本估计及限制。
12. UniSparse是否达到准入门槛。
13. 需要理论研究员进一步分析的具体公式及机制。
14. 全部异常和阻塞。
15. 是否存在来源可验证的预训练权重。

本轮任务是完成新的科研Baseline准入判断，不是为了维持科研循环而仓促启动新算法训练。

获得用户方向调整及资源授权后按协议执行。