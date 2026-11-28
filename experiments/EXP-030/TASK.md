# Codex科研执行任务：ROUND-028

# PD-TKD最后一次有限独立修复——阶段A：科研验收合同与完整训练准入

## 1. 任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-028
- 实验编号：EXP-030
- 科研决策：DEC-028，REPAIR + INDEPENDENT ADMISSION
- 理论版本：v0.3.3，保持不变
- 建议工程协议版本：v1.1.0
- 科学问题：Q-016
- 科研假设：H-030、H-031、H-032
- 候选方法：PD-TKD
- GitHub仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-027-pruning-damage-distillation
- 本轮建议分支：research/round-028-independent-admission
- 上轮预检代码SHA：9bec4165be0af584a75a198d1ab675df6deb78f1
- 上轮错误接受的训练锁及实际训练代码SHA：852a8b43f148bccfcdd718d1be3cb2933f157c64
- 上轮最终结果SHA：0d4cb3dcd88aeed1e15ba77be64ace3d30ad08af
- 依赖实验：EXP-018、EXP-024、EXP-027、EXP-028、EXP-029

P0：冻结原科研协议。

P0：建立独立验收合同，修复准入测试自我验证问题。

P0：重新执行完整真实训练预检。

P1：提交可供理论研究员独立审核的真实证据。

本次授权只允许阶段A。

绝对禁止启动正式训练、正式终点评价或自动执行后续阶段。

## 2. 必须读取的固定科研协议

完整原任务文件已经由理论研究员核验存在：

固定Commit：

0d4cb3dcd88aeed1e15ba77be64ace3d30ad08af

文件：

research/ROUND-027-TASK.md

以及：

experiments/EXP-029/TASK.md

两个文件的Git blob SHA均为：

afbd47e1dc48733c044c55cb893359443bf9793a

必须完整读取上述原任务。

可使用：

git show 0d4cb3dcd88aeed1e15ba77be64ace3d30ad08af:research/ROUND-027-TASK.md

核验文件身份：

git rev-parse 0d4cb3dcd88aeed1e15ba77be64ace3d30ad08af:research/ROUND-027-TASK.md

原任务中全部数学公式、固定模型、训练数据、实验分组、超参数、实验门槛、统计方法和资源限制继续有效。

本任务仅修改工程准入、执行阶段和授权机制。

如原任务与本任务的执行阶段授权冲突，以本任务更严格的限制为准。

不得修改原科学假设或性能门槛。

## 3. EXP-029科研归档

EXP-029：

执行状态PARTIAL。

训练证据INVALID_EVIDENCE。

原训练准入锁错误接受，永久保留历史无效状态。

实际完成：

K0：5轮。

K1：5轮。

K2：5轮。

K3：1轮，第2轮中断。

M6：0轮。

未执行正式验证集评价、最终统计及性能门槛判断。

H-030、H-031、H-032保持UNTESTED。

事后CPU人造矩阵检查及恢复比较器测试不得追认为原训练前检查。

本轮不得使用EXP-029任何部分训练模型作为新实验正式候选。

历史模型、训练日志、原锁及失败记录均保持只读。

保留EXP-027及EXP-028有效结果。

## 4. 已核验的科学机制证据

原D_grad预检：

样本数512。

非均匀时间权重样本比例1.0。

循环错位后不同权重元素2037。

正损伤时间步1875。

权重范围[0.5,2.5]。

平均权重最大偏差1.1920928955078125e-7。

R127最大归一化误差2.76e-7。

R128最大归一化误差4.41e-7。

R129最大归一化误差6.80e-7。

真实保留神经元脉冲不一致数量0。

这些结果仅提供真实网络机制证据，不能替代正式性能实验。

本轮仍需在独立新Run ID中重新执行这些必要检查。

不要求新结果与历史数字完全一致。

## 5. 原方法及实验配置锁定

保留原R127—R134。

B0教师SHA256：

cd6e151efe68ee77ef57ce474d0e83737321f271d4bace1ef8809013a66ae60d

P0学生初始模型SHA256：

95073e1c9b3731a792fc72d596b3fa1c96b86989b328241b00bb663020c80958

模型路径从原固定任务及有效检查点清单读取。

CIFAR-10。

原45000训练样本。

4500开发验证样本。

原固定D_grad 512张。

T=4。

仅训练fc2.weight和fc2.bias。

五组：

K0：CE。

K1：CE+时间平均KD。

K2：CE+均匀逐时间步KD。

K3：CE+循环错位损伤权重KD。

M6：CE+真实损伤权重KD。

rho=0.5。

beta=0.5。

蒸馏温度=4。

原LIF、CuPy后端和有效权重参数化保持不变。

原软阈值固定。

手动稀疏更新次数0。

正式训练配置仍为每组5轮，Batch Size=256，SGD，momentum=0.9，weight_decay=0，初始lr=0.01，原5轮cosine调度及seed=12450。

上述配置本阶段仅用于预检，禁止启动正式训练。

## 6. 理论研究员发现的剩余工程问题

已读取固定版本：

experiments/ROUND-027/gate.py

experiments/ROUND-027/preflight.py

experiments/ROUND-027/train.py

experiments/ROUND-027/endpoint_admission.py

experiments/ROUND-027/evaluate.py

原代码虽然在停止后增加了synthetic_readout_127_129和optimizer_recovery_exact，但仍存在以下风险：

第一，预期样本数量由实际测试记录派生，再写入准入锁。

第二，实际训练入口从准入锁自身读取expected_samples。

第三，gate.verify主要核查状态、字段及数量，尚不足以逐项验证证据内容。

第四，准入锁记录code_sha，但实际入口尚需明确验证正在执行的源码是否为经过验收的版本。

第五，既有负面测试尚需专门覆盖本轮遗漏的两个检查、覆盖范围共同错误及证据内容错误等情况。

必须针对这些问题进行最小修改。

不得仅增加若干PASSED字段。

不得修改历史ROUND-027代码及原实验记录。

新实现置于独立ROUND-028目录。

## 7. 建立独立科研验收合同

在执行测试之前，新建：

research/ROUND-028-ADMISSION-CONTRACT.json

以及可读版本：

research/ROUND-028-ADMISSION-CONTRACT.md

验收合同必须来源于原固定科研协议及本任务，不得来源于已经执行的测试结果。

每项必需测试至少定义：

contract_id。

stage。

target_groups。

required_input_identity。

expected_sample_count。

expected_input_ids_sha256（适用时）。

required_tensor_shapes。

required_assertions。

numeric_tolerance。

required_evidence_files。

required_code_identity。

failure_action。

必须在CPU预检及真实GPU预检开始前提交GitHub。

记录合同Commit SHA、文件SHA256及Git blob SHA。

后续禁止根据实际测试结果修改合同。

如发现合同歧义，停止并报告。

合同本身不包含任何尚未执行的测试结果。

## 8. 独立验收合同最低覆盖要求

不得将下面的项目合并成一个没有子项证据的布尔值。

### A组：身份与数据

A01：B0/P0固定文件哈希。

A02：真实模型结构及注册引用。

A03：固定数据划分和D_grad索引哈希。

A04：五组实际初始模型状态一致。

A05：五组优化器均仅含fc2.weight及fc2.bias。

A06：固定训练配置及学习率序列。

### B组：数学公式

B01：独立人造矩阵R127。

B02：独立人造矩阵R128。

B03：独立人造矩阵R129。

B04：真实网络R127，固定512张D_grad。

B05：真实网络R128，固定512张D_grad。

B06：真实网络R129，固定512张D_grad。

B07：真实教师/P0保留神经元逐元素脉冲一致性。

B08：零损伤权重退化。

B09：单时间步损伤。

B10：相同时间损伤。

B11：教师错误时间步处理。

B12：权重范围及时间平均为1。

B13：rho=0退化为均匀权重。

B14：K3循环错位及权重多重集合保护。

B15：教师学生相同logits时KD为0，允许原数值容差。

B16：K1/K2/K3/M6各自KD公式与独立FP64参考。

B17：真正2+1非等尾批。

B18：固定512张D_grad的时间权重非退化门槛。

B04—B07和B18的预期样本数必须直接在合同中固定为512，且使用原D_grad索引哈希。

不得在测试结束后根据实际读取数量生成预期值。

### C组：真实临时训练候选

对K0、K1、K2、K3、M6分别检查：

真实256张训练batch。

实际增强输入一致。

真实AMP前向与反向。

损失有限。

梯度有限及非零。

成功优化器更新。

分类读出更新。

全部非目标参数及buffer保护。

软阈值固定。

教师和P0无梯度及持久化状态变化。

原稀疏机制不变。

手动稀疏更新为0。

真实冻结特征脉冲一致性。

真实模型参数量11399524。

上述项目必须针对五组分别绑定实际候选证据。

### D组：完整检查点恢复

五组分别执行。

D01：真实临时模型完成优化器更新后保存检查点。

D02：真实恢复模型的全部参数及持久化buffer一致。

D03：优化器param_groups逐项比较。

D04：优化器全部state字典逐项比较。

D05：所有实际momentum_buffer张量的dtype、shape及数值逐元素一致。

D06：scheduler完整状态一致。

D07：GradScaler完整状态一致。

D08：使用相同固定输入及受控随机状态，对连续更新与保存恢复后更新执行一次真实对照。

D08属于本轮额外的工程恢复验收，不改变正式科研算法。

若原后端存在不能满足精确更新一致性的确定性限制，如实报告并暂停，不擅自放宽标准。

对D05不能只检查state条目数量。

检查必须确认实际已生成非空动量张量，不能以两个空字典相等作为通过。

保存每个参数对应的动量张量名称、shape、dtype、最大差及哈希。

### E组：资源

真实连续batch资源测量。

区分冷启动与稳定batch。

确认五组正式训练、必要评价和后续检查的保守资源估计。

本轮只测量，不启动正式训练。

### F组：负面拒绝测试

详见下一节。

## 9. 负面拒绝测试

在原十项负面测试基础上至少新增：

1. 单独缺少B01、B02或B03时拒绝。

2. 缺少真实优化器动量检查时拒绝。

3. 修改一个实际momentum_buffer元素时，恢复检查必须失败。

4. 实际和记录中的预期样本数量同时被改成256，但原合同要求512时，必须拒绝。

5. 样本数相同但样本索引哈希错误时，必须拒绝。

6. 某项记录标记PASSED，但真实误差超过合同容差时，必须拒绝。

7. 必需证据文件缺失或哈希不一致时，必须拒绝。

8. 实际训练代码或依赖源码发生未授权修改时，必须拒绝。

9. 测试证据针对错误候选模型时，必须拒绝。

10. 使用EXP-029旧锁时，必须拒绝。

11. 缺少阶段B独立用户授权记录时，必须拒绝正式训练。

12. 缺少终点人工审核授权时，必须拒绝正式评价。

对入口测试必须通过实际正式入口的守卫逻辑执行。

使用测试替身监测真实训练、优化器正式更新、正式验证加载器和评价调用次数。

准入失败时，上述正式操作次数必须为0。

不能仅测试独立validator返回False。

## 10. 独立证据验证机制

新的验证器必须以预先提交的科研验收合同作为唯一必需检查来源。

不能从准入结果中的required_tests、expected_samples等字段重新定义必需项目。

针对每项测试：

核对合同身份。

核对实际目标模型及数据身份。

核对实际测试代码。

核对证据文件哈希。

重新读取关键数值证据并判断是否满足合同断言。

核对实际退出状态。

确认没有缺失子项。

验证器必须拒绝缺失记录、重复记录、错误目标、错误样本、无效证据和未授权配置变化。

必要测试全部通过后才能生成候选训练准入锁。

该锁不等于正式训练授权。

## 11. 代码身份验证

记录所有实际执行及导入的项目科研代码文件。

包括：

ROUND-028新代码。

从历史实验导入的模型及数据加载工具。

固定UniSparse官方实现。

实际使用的兼容性补丁。

保存文件路径、Git blob SHA及实际文件SHA256。

准入审核后若上述文件发生变化，原对应准入锁失效。

不允许简单以当前HEAD等于旧预检Commit作为唯一代码身份验证，因为提交锁和报告可能正常改变HEAD。

应独立核查被验收的源码文件清单及哈希。

不提交依赖环境中的秘密信息。

## 12. 真实预检执行顺序

第一步：

检查Git状态、AGENTS.md和全部固定科研文件。

第二步：

归档历史EXP-029科学状态及当前任务。

第三步：

编写并提交独立验收合同。

第四步：

CPU执行人造矩阵检查、恢复比较器正反测试及所有入口负面测试。

第五步：

确认资源条件及本轮用户授权。

第六步：

使用一张GPU重新执行全部真实网络和五组临时候选预检。

第七步：

使用独立合同重新验收全部证据。

第八步：

生成候选训练准入锁、覆盖矩阵及完整报告。

第九步：

提交GitHub并核验远程结果。

第十步：

停止所有本轮执行进程。

等待理论研究员审核。

禁止自动启动正式训练。

## 13. 阶段A最终状态

全部合同检查通过时，只能记录：

execution_status=PREFLIGHT_COMPLETED

admission_status=PENDING_INDEPENDENT_REVIEW

formal_training_authorized=false

formal_training_started=false

不能将EXP-030整个实验标记COMPLETED。

不能记录正式训练组已完成。

不能生成虚构正式验证结果。

若检查失败：

记录PREFLIGHT_FAILED或BLOCKED，以及具体未满足合同条目。

保留全部原始证据和失败Run ID。

## 14. 后续阶段B执行约束

以下内容仅作为未来计划，当前不授权执行。

获得理论研究员独立审核通过及用户新授权后：

五组全部从原固定P0重新初始化。

不得从EXP-029的K0/K1/K2/K3继续训练。

不得使用EXP-029候选权重作为正式证据。

各5轮。

固定原训练协议、原R127—R134、原超参数。

训练结束后独立执行终点准入。

终点准入必须包含全部实际候选检查点身份、完整训练记录、冻结参数、保留神经元脉冲一致性及恢复模型/推理模型等价性。

正式评价前，另行提交真实终点评价锁及报告，等待审核授权。

只有正式训练和终点准入全部符合合同后，才能评价固定4500张验证样本。

保持原H-030/H-031/H-032及M6整体门槛：

H-030：K1-K0至少+0.3 pp。

H-031：K2-K1至少+0.3 pp。

H-032：M6-K2及M6-K3分别至少+0.3 pp。

M6整体条件还包括：

相对K0、K1分别至少+0.3 pp。

相对B0准确率损失不超过0.5 pp。

CE不高于K2。

物理参数量11399524。

SOP不超过P0的101%。

全部原科研与工程条件有效。

不搜索任何新参数。

不增加新算法。

## 15. 本阶段资源限制

需要用户授权。

单张符合实验室共享管理规定的RTX4090。

GPU总占用上限5分钟。

CPU累计计算上限30分钟。

正式训练Epoch=0。

正式验证集评价次数=0。

允许必要的真实临时模型更新及恢复对照。

所有前置检查、失败测试、资源测量、重试及真实预检计入预算。

CPU计量尽可能从本轮工作开始覆盖进程树。

不得用未经测量的资源时间冒充实测数据。

不干扰其他用户进程。

不启动后台正式训练。

资源不足时停止并报告。

## 16. 停止条件

任一以下条件发生时，不允许进入下一阶段：

原固定科学协议身份无法核验。

独立验收合同未提前提交。

必需检查覆盖不足。

真实公式检查超限。

优化器完整状态恢复失败。

实际模型或数据身份不一致。

真实候选保护失败。

原软阈值变化。

出现未授权手动稀疏更新。

代码身份不一致。

真实入口负面拒绝测试失败。

资源不足或超限。

未获得独立人工审核及后续授权。

不允许通过减少测试项目、修改合同或降低容差取得形式上的通过。

## 17. 科研状态更新

新增：

research/ROUND-028-TASK.md

research/ROUND-028-ADMISSION-CONTRACT.json

research/ROUND-028-ADMISSION-CONTRACT.md

更新：

AGENTS.md。

research/RESEARCH_STATE.md。

research/HYPOTHESES.md。

research/EXPERIMENT_PLAN.md。

research/EXPERIMENT_REGISTRY.csv。

research/DECISION_LOG.md。

AGENTS.md追加：

重要实验的必需检查与预期覆盖必须先由独立合同定义。

禁止从实际测试结果推导必需检查或预期覆盖。

正式入口须重新核查证据内容和代码身份。

预检完成不等于正式训练授权。

涉及此前重复准入失效的实验，关键阶段必须经过独立审查。

理论版本保持v0.3.3，不新增科学公式。

EXP-029永久PARTIAL / INVALID_EVIDENCE。

EXP-030按阶段实际状态记录。

H-030/H-031/H-032保持UNTESTED。

## 18. 阶段A交付文件

新建独立目录：

experiments/ROUND-028/

experiments/EXP-030/

至少保存：

TASK.md

config.json

environment.txt

code_version.json

source_identity_manifest.json

commands.sh

status.json

contract_identity.json

protocol_requirement_traceability.csv

parent_checkpoint_audit.json

data_split_audit.json

synthetic_readout_tests.json

temporal_readout_audit.json

formula_tests.json

damage_summary.json

weight_diagnostics.csv

candidate_preflight_results.json

optimizer_recovery_exact.json

optimizer_momentum_tensor_audit.csv

recovery_update_equivalence.json

teacher_protection_audit.json

frozen_parameter_audit.json

admission_negative_tests.json

admission_coverage_matrix.csv

independent_contract_verification.json

candidate_training_admission_lock.json

resource_benchmark.json

resource_usage.csv

REPORT.md

limitations.md

run.log

其中protocol_requirement_traceability.csv必须逐项对应原科研协议、独立合同、实际测试函数及真实证据。

不能只有测试名称和PASSED字段。

历史失败文件保持不变。

本阶段未产生正式训练指标、终点检查点或验证结果，不得虚构。

## 19. GitHub交付与强制停止

从上轮最终结果SHA建立独立分支。

先提交独立验收合同。

再提交实际修复代码。

执行预检。

最后提交全部真实证据和候选准入锁。

不修改main，不force push。

返回：

- GitHub实验分支。
- 独立验收合同Commit SHA。
- 实际预检代码SHA。
- 候选训练准入锁SHA。
- 最终结果提交SHA。
- EXP-030阶段A执行状态。
- 原协议必需检查覆盖情况。
- 两项历史遗漏检查的新结果。
- 五组实际优化器动量恢复结果。
- 连续与恢复更新对照结果。
- 全部真实入口负面测试。
- 资源基准和实际资源消耗。
- 报告及原始证据路径。
- 尚未通过或无法确认的项目。

必须明确报告：

正式训练是否启动。

正式验证集是否读取。

是否存在后台训练进程。

正常情况下三项均为否。

提交完成后停止。

本任务不允许自动进入阶段B，不允许自行追加正式训练。

等待理论研究员读取真实GitHub内容、审核并由用户给予新的资源授权。