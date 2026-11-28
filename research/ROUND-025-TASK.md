# Codex 科研执行任务：ROUND-025

# UniSparse高比例结构化剪枝独立准入修复与固定候选性能复核

## 1. 任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-025
- 实验编号：EXP-027
- 科研决策：DEC-025，REPAIR + REPLICATE
- 理论版本：v0.3.1
- 科学问题：Q-014
- 科研假设：H-026、H-027
- 项目仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-024-structural-reconstruction
- 本轮建议分支：research/round-025-admission-replication
- 上轮实际运行代码SHA：9a08c3d634d780b7557b65ace838d7b809dbe481
- 上轮选择锁SHA：850e91413006d5cac4c8394dcd557057a7273f86
- 上轮最终结果SHA：7f86ebc8d9ee8a9b9cc0595fe570dbe2df1e23ad
- UniSparse官方固定SHA：b5b3da4fcd2e7abcd2c20c93d41cf63ef3201daa
- 依赖实验：EXP-018、EXP-024、EXP-026

P0：修复实验准入覆盖及状态硬编码问题。

P0：验证真实最终锁定候选模型。

P1：在完整准入后，独立重新评价五个固定模型。

本轮不是新的创新方法设计任务。

禁止重新拟合、改变剪枝集合、修改正则化系数、增加训练、开发中心化重构方法或使用官方测试集。

## 2. ROUND-024正式科研判定

EXP-026原计算进程为COMPLETED。

正式实验状态永久保留INVALID。

原因：

原准入程序仅在人工指定的1024神经元删除集合及人工补偿模型上验证R120真实脉冲一致性。

最终选出的50%和75% R1/R3模型未在正式评价前完成独立逐元素脉冲比对。

不得通过事后补测，将原EXP-026重新标记为有效。

原始计算结果、自动门槛、独立偏差审计及修正门槛全部保留。

H-026/H-027当前科学状态记录为INVALID_EVIDENCE或科研状态规范中等价的“实验无效，无法判断”。

保留原UNTESTED历史记录。

本轮通过独立EXP-027重新获得有效证据。

## 3. EXP-026探索性结果

严格FP32 B0：

- Accuracy=94.222222%。
- CE=0.1727328626844618。
- SOP/sample=86009551.08977778。

50% L1，R1：

- Accuracy=93.933333%。
- CE=0.1793660873969396。
- SOP/sample=78340358.21466666。
- 参数量19839844。

50%重构，R3：

- Accuracy=93.911111%。
- CE=0.1873258376121521。
- SOP/sample=78340358.21466666。
- 参数量19839844。

75% L1，R1：

- Accuracy=92.511111%。
- CE=0.27720849974950157。
- SOP/sample=74488295.33955556。
- 参数量11399524。

75%重构，R3：

- Accuracy=90.733333%。
- CE=0.32743940374586317。
- SOP/sample=74488295.33955556。
- 参数量11399524。

D_cal logits MSE：

50% R1：6.646323382930275。

50% R3：0.4444129560858133。

75% R1：20.662515923714174。

75% R3：4.849417241073305。

这些结果只能作为历史探索性参考。

本轮实际结果不要求与之完全一致。

若存在数值差异，必须如实报告和分析，不修改新结果以匹配旧结果。

## 4. 工程问题定位

理论研究员已读取实际运行版本：

experiments/ROUND-024/run.py

第59行附近：

R120真实网络测试只使用人工指定删除集合。

第66—68行附近：

原准入通过依据测试状态及数量，没有检查最终候选覆盖。

第105行附近：

admission_and_isolation直接硬编码True。

本轮必须解决上述三个问题。

不得仅修改最终报告中的PASSED或FAILED文字。

不得修改历史ROUND-024代码及结果。

在独立ROUND-025代码目录实现新准入机制。

## 5. 固定父检查点及模型

父检查点：

experiments/EXP-018/runs/EXP-018-CONTINUE-001/U2/epoch-300.pth

SHA256：

19b1312cbaae49ad95a7d001270c4357d7ea23b2584d19698b0084b3da4b094f

原模型清单：

experiments/EXP-026/checkpoint_manifest.json

本轮只评价：

B0。

50-R1。

50-R3。

75-R1。

75-R3。

优先直接加载原清单中的真实模型文件。

固定模型SHA256：

B0：

cd6e151efe68ee77ef57ce474d0e83737321f271d4bace1ef8809013a66ae60d

50-R1：

21ad97345e5e738417dd4d0d43439e4cf437c8d171fc4a7acdeec5f827200c3c

50-R3：

c653b388bf65511fe4f38198b75fe2ce451a9b89ef511d5aa26fbe79f3aea94b

75-R1：

95073e1c9b3731a792fc72d596b3fa1c96b86989b328241b00bb663020c80958

75-R3：

7c885e34b59f506dd723b862a7012cfd70974952c65c1e1b719693ff5969cf57

实际路径从原manifest读取。

若文件缺失或哈希不符，停止对应实验，报告阻塞。

不得重新拟合替代缺失的R3。

所有原检查点只读。

运行前后核对哈希。

## 6. 原选择锁核验

从原选择锁提交读取：

experiments/EXP-026/selected_neurons.json

SHA256：

56e4d123717a6937ee9b0f9772821c99e266239bcd356a115675643cd21a0736

experiments/EXP-026/reconstruction_coefficients.csv

SHA256：

299aeeab0d9d81baca47d14f93e42c6c2f5bb86e21c750b8b073a087d327e13f

experiments/EXP-026/readout_support_audit.json

SHA256：

ef65463d0f45985e0bb3dad7506bfbb1e564e71467df8b9a5fd10e0efe37f2d3

核查原选择锁文件及远程提交。

检查实际模型与锁定保留索引、原软阈值参数及补偿权重一致。

不得重新排序神经元。

不得重新求解ridge系统。

不得改变原lambda_scale=0.001。

本轮只验证已经锁定的候选。

## 7. 固定实验环境

沿用EXP-024 RUN-002及EXP-026 RUN-002：

- CIFAR-10。
- CifarNet。
- base_channels=256。
- T=4。
- 官方CuPy LIF。
- 严格FP32。
- 禁用cuDNN TF32。
- 禁用matmul TF32。
- 禁用AMP。
- Batch Size=64。
- workers=4。
- 确定性评价预处理。
- model.eval()。
- 所有前向前后正确reset神经元状态。

不使用优化器。

不改变训练配置或模型动力学。

## 8. 数据协议

继续使用：

experiments/EXP-004/splits/indices.json

SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

D_grad：

512张，使用原固定索引。

索引哈希：

93c4c834c3fd3c8b92a64caa88da8a671e27be23dc2df11999c8e5f3dfbe47d8

D_cal：

512张，使用原固定索引。

索引哈希：

b671cbd803b13cc76b02f5db4bad311ef9492276622051df01af56a931301baa

固定验证集4500张。

D_grad用于实际候选准入及后续诊断。

只有实际候选完整准入通过后，才能创建或迭代D_cal及正式验证评价入口。

不读取官方test_batch。

D_grad/D_cal均来自原父模型训练集。

固定开发验证集已多次使用。

本轮属于独立工程复核，不是独立训练种子或独立测试集确认实验。

## 9. 两阶段准入机制

阶段A：通用正确性检查。

包括：

- 父检查点与模型哈希。
- 实际结构与参数数量。
- 数据隔离。
- 严格FP32设置。
- R103读出重构。
- R120补偿权重映射。
- 真正2+1非等尾批。
- 神经元reset。
- 捕获开启/关闭一致性。
- 有效连接支持集审计。

通用测试通过只能标记：

GENERIC_PREFLIGHT_PASSED。

不能标记最终实验准入通过。

阶段B：真实候选模型检查。

必须针对：

50-R1 vs 50-R3。

75-R1 vs 75-R3。

逐一独立执行。

人工固定删除集合或人工补偿模型不能替代真实候选检查。

## 10. 实际候选脉冲一致性

对50%和75%两个真实模型对，使用全部512张固定D_grad样本。

捕获最后隐层activation_fc1真实输出：

50%：

s.shape=[T,B,1024,1,1]。

75%：

s.shape=[T,B,512,1,1]。

同一模型对必须使用完全相同的实际输入和样本顺序。

每批独立复位神经元。

对R1和R3的真实脉冲张量执行：

torch.equal(s_R1,s_R3)

另外记录：

mismatch_count=sum(s_R1!=s_R3)

max_abs_difference=max(abs(s_R1-s_R3))

检查脉冲形状、数据类型及二值性。

必须满足：

全部实际样本覆盖。

mismatch_count=0。

max_abs_difference=0。

不能使用FR、SOP、时间平均脉冲数量相同替代逐元素相等。

逐Batch记录：

- 实际样本ID。
- 实际样本数量。
- 模型哈希。
- 实际捕获模块。
- 脉冲张量shape。
- 实际比较元素数量。
- 不一致元素数量。
- 最大绝对差。
- 两组脉冲张量哈希。
- 检查退出状态。

如任一批失败，停止正式评价。

## 11. 真实候选结构及参数审计

对于两个比例：

确认R1/R3保留神经元索引完全相同。

确认物理参数量相同。

确认除fc2潜在权重外，其他模型参数及持久化buffer完全一致。

确认fc2阈值及bias保持不变。

确认R3有效掩码是R1有效掩码的子集。

新增有效连接数量必须为0。

确认R3有效读出权重符合锁定的原R120补偿。

对真实模型logits核对：

z_R3-z_R1=R_retained@Delta_A.T

其中R_retained来自真实脉冲时间平均。

Delta_A由实际有效fc2权重变化计算。

归一化误差门槛保持1e-5。

使用真实模型前向及FP64离线代数重构。

检查全部D_grad样本。

不得通过放宽容差获得通过。

## 12. 原父模型与紧凑模型等价性

对实际50-R1及75-R1删除集合：

使用固定D_grad前32张样本。

验证真实紧凑模型与原父模型对应零脉冲门控结果。

检查保留神经元脉冲逐元素一致。

检查真实紧凑模型logits与门控参考在原容差内一致。

检查分类预测及CE。

需要确认两种不同删除集合，而不是只检查一个比例。

## 13. R122机器可核验准入锁

为正式评价定义明确的必需测试清单。

每个测试记录必须至少包含：

test_id。

target_group。

target_model_sha256。

target_pair_sha256（如适用）。

input_ids_sha256。

actual_sample_count。

expected_sample_count。

assertion。

observed_result。

status。

actual_exit_code。

required_tests必须覆盖实际两个比例及全部必要检查。

passed_tests仅包含已真实运行且所有断言通过的记录。

只有required_tests是passed_tests的子集，且模型身份、输入覆盖、实际退出状态均一致时，才允许正式评价。

禁止：

admission_and_isolation=True

或其他形式的硬编码准入状态。

禁止依据测试数量相同判定覆盖完整。

禁止使用预先填写的True代替真实断言结果。

新增负面单元测试：

1. 缺少50%实际候选检查时，必须拒绝。
2. 缺少75%实际候选检查时，必须拒绝。
3. 只有人工测试通过时，必须拒绝。
4. 候选模型哈希不一致时，必须拒绝。
5. 样本覆盖不足时，必须拒绝。
6. 测试断言失败或退出状态异常时，必须拒绝。
7. 准入锁创建后候选模型改变时，必须拒绝。

准入代码必须在正式评价入口实际执行。

不能只在独立测试文件中存在而不影响正式评价。

## 14. 正式评价前锁定

完整准入通过后生成：

candidate_admission_lock.json

包括：

原模型清单哈希。

实际五个模型SHA256。

原选择锁SHA。

原神经元索引及补偿数据哈希。

全部通用测试结果。

全部真实候选测试结果。

实际测试覆盖矩阵。

必需测试与通过测试集合。

数据隔离审计。

固定配置及原判定标准。

实际代码SHA。

将准入锁提交GitHub。

核实远程提交成功后，才允许进入D_cal及验证评价阶段。

若无法完成远程锁定，报告阻塞。

不将EXP-026历史选择锁视作本轮最终准入锁。

## 15. 科学公式

复用R101—R120，不改变既有方法。

新增理论记录R121：

对logits误差e：

mean_e=mean(e,dim=-1)

e_centered=e-mean_e

sum(e²)=sum(e_centered²)+K*mean_e²

其中K=10。

该分解是精确代数关系。

公共logits偏移不影响softmax及交叉熵。

本轮允许在已经采集的D_grad和D_cal真实logits上计算：

MSE_raw=mean((z_teacher-z_candidate)²)

MSE_centered=mean((e-mean(e,dim=-1))²)

MSE_shift=mean(mean(e,dim=-1)²)

核对：

MSE_raw=MSE_centered+MSE_shift

使用FP64统计。

该分析仅为探索性诊断。

不改变既有补偿模型、拟合目标或主要判定规则。

不实现或拟合中心化logits重构新方法。

## 16. 正式独立评价

准入锁通过后，对五个固定模型重新执行：

B0。

50-R1。

50-R3。

75-R1。

75-R3。

完整4500张验证样本。

对R1/R3另执行原D_grad和D_cal重构评价。

全部记录使用本轮真实计算结果。

不得直接复制EXP-026验证指标作为本轮实验结果。

保存：

Accuracy。

CE。

FR_global。

SOP_sample_weighted。

SOP_author。

逐层SOP。

实际物理参数量。

有效连接数量及稀疏率。

逐样本预测、标签、正确性及CE。

教师logits MSE、RMSE及预测一致率。

中心化MSE及公共偏移MSE。

B0对比有效EXP-024历史结果。

各组对比EXP-026历史探索性结果。

差异必须如实报告。

## 17. 原科研假设及判定规则

不修改H-026/H-027原门槛。

H-026主要条件：

50%剪枝下：

Accuracy_B0-Accuracy_R1>=1个百分点。

75%仍为辅助探索。

不得用75%的结果替代50%的主要条件。

H-027主要条件：

固定50%剪枝下：

1. R3的D_cal教师logits MSE低于R1。
2. R3验证准确率比R1提高至少0.5个百分点。
3. R3验证CE低于R1。
4. R3和R1物理参数量相同。
5. R3有效连接支持集不扩大。
6. R3实际SOP不超过R1的101%。
7. 完整准入及数据隔离通过。

75%按相同指标单独报告，不改变主要条件。

若复核后依然未达到原门槛，记录有效负面证据。

不得追加调整lambda、拟合样本数、模型结构、剪枝率或补偿方式。

Codex仅报告门槛结果，不擅自修改科学假设最终状态。

## 18. 配对统计

使用4500张固定验证样本。

固定2000次配对Bootstrap。

seed=20261128。

numpy.default_rng PCG64。

百分位95%区间。

比较：

50-R1 minus B0。

50-R3 minus 50-R1。

75-R1 minus B0。

75-R3 minus 75-R1。

报告准确率百分点差、CE差、配对置信区间及预测差异。

若没有逐样本SOP，则对应配对区间标记NA。

本轮没有新增训练种子。

不能将同一模型的重复评价解释为独立训练重复实验。

## 19. 资源预算

需要用户重新授权。

建议：

一张符合实验室共享规定的RTX4090。

GPU总占用上限10分钟。

CPU累计计算上限30分钟。

新增训练Epoch=0。

禁止第二张GPU。

所有准入检查、失败重试、诊断、评价及报告生成计入预算。

从本轮开始尽可能完整记录CPU进程树及保守资源预算。

实际GPU进程墙钟与GPU内核执行时间严格区分。

不得干扰其他用户进程。

## 20. 停止条件

以下任一情况发生时停止相应正式任务：

- 模型或选择锁哈希不一致。
- 已保存模型无法正确加载。
- 实际候选脉冲不一致。
- 实际候选覆盖不完整。
- 非目标参数变化。
- 新增有效连接。
- 读出补偿重构失败。
- 数据隔离失败。
- 必要准入锁未提交。
- 非有限数值。
- OOM。
- 资源超限。

不得使用其他人工候选替代真实候选。

不得为了完成本轮任务而绕过准入失败。

## 21. AGENTS.md长期规范更新

追加以下规定：

第一，所有真实候选均需要对象级准入证据。

第二，通用单元测试与实际候选测试分别管理。

第三，测试名称、实际输入及断言必须一致。

第四，候选哈希发生变化时，原对象级准入证据失效。

第五，正式评价入口必须主动验证准入锁。

第六，禁止硬编码科研门槛或准入状态。

第七，计算完成状态与实验有效性分别记录。

第八，事后补测不能恢复历史预评价准入合规性。

第九，独立复核使用新的实验编号或Run ID。

第十，必须保留全部历史失败记录。

不得删除原AGENTS.md规范。

## 22. 科研状态更新

新增：

research/ROUND-025-TASK.md

更新：

research/THEORY.md，追加v0.3.1及R121/R122。

research/HYPOTHESES.md。

research/RESEARCH_STATE.md。

research/EXPERIMENT_PLAN.md。

research/EXPERIMENT_REGISTRY.csv。

research/DECISION_LOG.md，追加DEC-025。

AGENTS.md。

EXP-026保持INVALID。

EXP-027独立记录实际状态。

UniSparse保持Controlled Baseline。

候选Baseline暂不切换。

固定支持集重构方法暂停推进，等待本轮最终审查。

不创建额外创新候选。

## 23. 实验交付

新增目录：

experiments/ROUND-025/

experiments/EXP-027/

至少保存：

- TASK.md
- config.json
- environment.txt
- code_version.json
- commands.sh
- status.json
- parent_checkpoint_audit.json
- fixed_model_manifest.json
- historical_selection_lock_audit.json
- formula_tests.json
- generic_preflight_audit.json
- candidate_spike_equivalence.csv
- candidate_parameter_audit.json
- actual_candidate_readout_audit.json
- structural_equivalence_tests.json
- admission_coverage_matrix.csv
- admission_negative_tests.json
- candidate_admission_lock.json
- admission_lock_commit.json
- reconstruction_summary.csv
- logit_error_decomposition.csv
- validation_summary.csv
- validation_per_sample.csv
- historical_reproduction_audit.json
- layer_sops.csv
- paired_bootstrap.json
- hypothesis_gate_results.json
- resource_usage.csv
- REPORT.md
- limitations.md
- run.log

未执行部分明确记录原因。

不生成虚构数据。

保留失败Run ID及原始记录。

## 24. GitHub交付

从上轮最终结果SHA创建独立分支。

正式运行前提交代码和协议。

正式评价前提交完整真实候选准入锁。

完成后提交真实结果、科研状态及长期规范更新。

不修改main。

不force push。

不覆盖EXP-026。

最终返回：

- GitHub实验分支链接。
- 实际执行代码SHA。
- 候选准入锁SHA。
- 最终结果提交SHA。
- EXP-027正式执行及有效性状态。
- 两个真实候选模型对的脉冲一致性结果。
- 实际覆盖样本及张量元素数量。
- 全部准入负面测试结果。
- 五个模型真实Accuracy、CE、FR、SOP及参数量。
- R1/R3重构误差及中心化误差。
- 配对统计及H-026/H-027门槛结果。
- 与EXP-026探索性数据的实际差异。
- GPU及CPU资源消耗。
- 异常、阻塞和未完成项目。
- 实验报告及原始数据路径。

本轮仅进行一次范围固定的独立复核。

不开展新方法设计，不增加训练，不自动进入下一轮。

获得用户授权后执行。