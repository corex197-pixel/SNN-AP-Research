# Codex 科研执行任务：ROUND-019

# MP-EFR精确SOP预算匹配与公平性能验证

## 1. 任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-019
- 科研阶段：阶段1—3，候选方法机制与性能验证
- 科研决策：DEC-019，REFINE + CONTINUE
- 理论版本：v0.2.5
- 科学问题：Q-010
- 科研假设：H-016、H-017、H-018、H-019
- 实验编号：EXP-021
- 候选方法：MP-EFR（Mean-Preserving Effective Fan-out Regularization）
- 项目仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-018-fanout-regularization
- 本轮建议分支：research/round-019-matched-budget
- 上轮正式执行代码SHA：32848cf953f666372a45fc7771a0df59de73708b
- 上轮最终结果SHA：808b0ae54a1ac25c6b7d0fca2b745df23cf5f4f1
- UniSparse官方固定SHA：b5b3da4fcd2e7abcd2c20c93d41cf63ef3201daa
- 依赖实验：EXP-017、EXP-018、EXP-019、EXP-020

优先级：

P0：历史结果、父模型及原数学实现一致性核验。

P1：补齐G0/G2计算预算匹配。

P1：固定验证集公平性能比较。

本轮不修改MP-EFR数学机制，不开展SGD训练，不搜索rho，不使用官方测试集。

## 2. ROUND-018正式科研判定

EXP-020为COMPLETED。

整体推进门槛未通过，原因是G0/G2在原固定q集合中没有达到校准集4%—6%的预算区间，导致同预算性能比较无法执行。

保留原失败门槛，不事后修改EXP-020的判定。

H-019：

SUPPORTED，但严格限定于当前U2第300轮模型、固定两个目标模块、rho=0.25、预注册q=0.02、指定样本及单次eval局部干预。

H-017：

INSUFFICIENT_EVIDENCE。

固定模型上的初步结果为正向信号，但缺少公平预算对照。正式训练有效性仍为UNTESTED。

H-016保持SUPPORTED及原适用边界。

H-018保持PARTIALLY_SUPPORTED及原负面证据。

## 3. 上轮真实结果

EXP-020校准集B0：

- 样本数512。
- Accuracy=0.99609375。
- CE=0.01414713598205708。
- SOP_sample_weighted=86348901.18554688。
- 两个目标消费者SOP=34006691.125。

主要机制点q=0.02：

G0目标消费者SOP下降=1680901.625。

G1目标消费者SOP下降=1914396.875。

G2目标消费者SOP下降=1640347.25。

三组实际整体参数更新范数约为0.5780679。

G1相对G0的目标消费者SOP下降量提高约13.89%。

G1相对G2提高约16.71%。

G1在全部7个非零尺度上取得比G0/G2更大的全网络SOP下降。

但较大干预尺度下G1的校准准确率也可能明显下降。

不宣称已经获得精度—计算量权衡优势。

上一轮验证集：

B0：

- Accuracy=0.9424444444444444。
- CE=0.17519400440322028。
- FR=0.04957846704501978。
- SOP=86008931.65022223。

G1，q=0.04：

- Accuracy=0.9428888888888889。
- CE=0.17795569733778635。
- FR=0.04647743348817568。
- SOP=82161247.10466667。

G1相对B0：

- SOP下降约4.4736%。
- 准确率提高0.04444个百分点。
- 平均CE增加约0.002762。
- 权重掩码和非零有效权重数保持不变。

理论研究员已独立配对分析EXP-020的9000条验证记录：

- B0错误、G1正确：67张。
- B0正确、G1错误：65张。
- 净新增正确预测2张。
- 两组预测完全一致4353张。
- 预测类别不同147张。

独立探索性Bootstrap区间跨零。

不得宣称G1已显著提高准确率或已经证明无损。

## 4. 固定父模型

继续使用：

experiments/EXP-018/runs/EXP-018-CONTINUE-001/U2/epoch-300.pth

SHA256：

19b1312cbaae49ad95a7d001270c4357d7ea23b2584d19698b0084b3da4b094f

保持父检查点只读。

运行前后均核对哈希。

不使用best_validation模型。

不恢复优化器进行正式训练。

## 5. 复用原始实验实现

首先读取：

- AGENTS.md
- research/RESEARCH_STATE.md
- research/THEORY.md
- research/HYPOTHESES.md
- research/ROUND-018-TASK.md
- experiments/EXP-020/REPORT.md
- experiments/EXP-020/config.json
- experiments/EXP-020/data_split.json
- experiments/EXP-020/calibration_response.csv
- experiments/EXP-020/permutation.json
- experiments/EXP-020/selection_lock.json
- experiments/EXP-020/gradient_statistics.csv
- experiments/EXP-020/validation_per_sample.csv
- experiments/EXP-020/hypothesis_gate_results.json

复用：

experiments/ROUND-018/run.py

experiments/ROUND-018/math_core.py

注意原run.py包含EXP-020固定输出路径。

不得直接执行会覆盖EXP-020历史结果的原入口。

建议建立独立：

experiments/ROUND-019/

以及：

experiments/EXP-021/

通过最小代码适配复用已有公式、模型加载、梯度计算、参数干预、状态审计和SOP评价逻辑。

不修改vendor/UniSparse官方源文件。

不开展无关重构。

## 6. 固定数学公式

目标路径：

L1：

static_conv.norm -> static_conv.activation -> conv1.conv

L2：

conv1.norm -> conv1.activation -> conv2

仅干预两个目标BN的gamma和beta。

活动：

s.shape=[T,B,C,H,W]

r[c]=mean(s,dim=(0,1,3,4))

有效扇出：

d[c]=sum(M[:,c,:,:])

M为后继SparseConv2d的实际有效二值掩码。

d0=C_out*k_h*k_w

d_mean=mean(d)

R75：

a[c]=(1-rho)+rho*d[c]/d_mean

固定rho=0.25。

d_mean=0时回退全1。

M及a均detach。

R76：

P_MP=d0*sum_c(a[c]*r[c])

G0：

a[c]=1。

G1：

a[c]=0.75+0.25*d[c]/d_mean。

G2：

使用EXP-020保存的实际permutation，将G1系数按同样方式置换。

不重新选择随机置换。

R79：

g_tilde[m,l]=g[m,l]*norm(g[0,l])/norm(g[m,l])

每层分别匹配gamma和beta连接向量的L2范数。

R80：

eta(q)=q*norm(phi)/norm(g0)

phi_new[m,l]=phi_parent[l]-eta(q)*g_tilde[m,l]

每次干预从同一父模型开始。

不累积干预。

除q的预算匹配外，保持EXP-020全部科学含义。

## 7. 新增机制诊断指标R81

定义：

E_m(q)=(C_target_B0-C_target_m(q))/norm(Delta_phi_m(q))

其中：

C_target为两个目标消费者的实际SOP/sample之和。

Delta_phi为两个目标BN affine实际更新向量。

norm为L2范数。

q=0或实际更新范数为0时标记NA，不除零。

该指标仅用于同一模型、相同参数化条件下的局部活动优化效率分析。

不将R81作为训练损失，不将其解释为严格物理效率。

保留EXP-020预注册q=0.02作为H-019主要证据。

本轮不重新选择H-019主要尺度。

## 8. 固定数据及运行条件

严格复用EXP-020的实际样本索引。

D_grad：

512张。

索引记录哈希：

93c4c834c3fd3c8b92a64caa88da8a671e27be23dc2df11999c8e5f3dfbe47d8

D_cal：

512张。

索引记录哈希：

b671cbd803b13cc76b02f5db4bad311ef9492276622051df01af56a931301baa

按EXP-020相同哈希序列化方式核对，不重新采样。

验证集：

固定4500张。

原完整划分文件：

experiments/EXP-004/splits/indices.json

SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

D_grad和D_cal已参与父模型训练。

验证集已参与历史工作点选择和EXP-020评价。

本轮属于开发阶段的探索性比较，不是独立最终测试。

500张旧科研诊断池不使用。

禁止读取官方test_batch。

固定：

- CifarNet。
- base_channels=256。
- T=4。
- 官方CuPy LIF。
- 原专用环境。
- Batch Size=64。
- workers=4。
- seed=20261128。
- FP32。
- 确定性评价预处理。
- 无SGD。
- BN运行统计冻结。

## 9. 梯度重建与必要回归检查

按照EXP-020完全相同的eval BN局部重放方法，在D_grad重新计算局部梯度。

保持原LIF状态复位、替代梯度、Batch样本加权及参数化一致。

核对：

- 有效扇出向量。
- 通道惩罚系数。
- 固定随机置换。
- 原始梯度范数。
- 匹配后梯度范数。
- G1/G2相对G0余弦相似度。
- R78公式。
- R79范数匹配。
- 局部重放输出与真实完整网络一致。
- 权重及BN状态完整性。

与EXP-020真实梯度统计比较。

建议可复现性检查：已记录梯度范数及余弦统计的归一化误差不超过1e-5；若存在非确定性导致差异，报告具体值及其对结果的影响。

EXP-020未保存完整梯度张量，不能仅凭汇总统计一致宣称梯度逐元素一致。

本轮应保存两个目标层的小型原始及范数匹配后梯度向量，便于后续复查。

不需要重复整套历史工程测试，只进行必要回归。

## 10. EXP-021A：预算匹配

### 10.1 固定G1锚点

本轮G1严格固定：

q_G1=0.04。

不允许重新选择。

首先在D_cal重测B0及G1。

历史参考值：

C_B0=86348901.18554688。

C_G1=82539922.0390625。

历史G1降幅约4.4111%。

核对当前两组与历史成本、准确率、CE及状态一致性。

如果历史结果无法合理复现，停止正式预算匹配并报告。

新的匹配目标采用通过一致性审计后的本轮实际G1校准成本：

C_star=C_G1(q=0.04)

R_star=1-C_star/C_B0

该目标必须保持在原4%—6%预算区间内。

本轮使用G1锚点而非重新搜索G1最优工作点，避免根据上一轮验证数据进一步选择G1。

### 10.2 G0/G2初始区间

在D_cal上核查G0和G2：

q_a=0.04。

q_b=0.08。

两个边界必须存在实际有效测量，并包围R_star。

历史结果：

G0：

q=0.04时SOP=83025861.84960938。

q=0.08时SOP=79465899.18164062。

G2：

q=0.04时SOP=83109661.38671875。

q=0.08时SOP=79667880.5625。

### 10.3 R82插值

定义：

R(q)=1-C(q)/C_B0

q_new=q_a+(R_star-R(q_a))*(q_b-q_a)/(R(q_b)-R(q_a))

根据EXP-020历史数据：

G0建议首个插值点约0.0454600551。

G2建议首个插值点约0.0466214483。

实际候选以本轮重新测量后代入R82的计算值为准。

插值只是候选q的计算方式，不是实际SOP测量值。

必须运行真实完整模型前向。

### 10.4 有限补测规则

预设校准匹配容差：

abs(R_m(q)-R_star)<=0.001

即降幅差不超过0.1个百分点。

同时必须满足原4%—6%区间。

若首个插值点达到上述条件，立即锁定该组。

若未达到：

- 依据实际成本更新包围区间。
- 使用区间中点进行有限二分补测。
- 每组最多追加4次中点测试。
- 每个候选均从父模型重新干预。
- 不扩大初始区间[0.04,0.08]。

若出现不符合区间包围条件或明显非单调响应：

保留所有真实观测，不能将线性插值或二分法视为严格数学保证。

在已测试候选中按照纯成本误差选择；若仍无满足条件候选，标记预算匹配失败，不继续扩大搜索。

在多个满足条件的已测候选中，优先选择成本误差最小者；如完全相同，选择较小q。

禁止根据Accuracy、CE或任何验证集表现选择q。

本轮每组除边界复测外，最多1次插值点及4次二分补测。

## 11. 预算匹配锁定

完成校准后保存：

- 实际B0成本。
- 实际G1锚点成本及降幅。
- G0/G2每次候选q。
- 每次实际测得的SOP。
- 匹配误差。
- 区间更新记录。
- 最终候选q。
- 完整梯度向量哈希。
- 固定置换及惩罚系数哈希。

在最终验证开始前提交selection_lock.json及实际锁定配置。

G1始终为q=0.04。

不得在看到验证集结果后重新调整任何组的q。

若某对照组无法通过预算匹配，明确标记未匹配。

不因为部分组缺失而伪造完整公平比较。

## 12. EXP-021B：固定验证集评价

在相同FP32、Batch Size、SOP监控范围及评价条件下运行：

B0：原U2第300轮。

G0：本轮锁定q。

G1：固定q=0.04。

G2：本轮锁定q。

若预算匹配失败，保留已锁定组的结果，失败组不得作为主要匹配比较。

核查B0/G1与EXP-020逐样本预测、CE及SOP的一致性。

如有差异，报告真实差异和原因，不静默替换历史数据。

每组记录：

- 验证准确率。
- 平均CE。
- 全局加权FR。
- 各消费者输入活动率。
- 两个目标消费者合计SOP。
- 全网络SOP_sample_weighted。
- SOP_author。
- 有效权重稀疏率。
- 有效非零权重数量。
- 实际参数更新范数。
- 每个目标层实际更新范数。
- 逐样本预测、真实标签、正确性和CE。

建议同时保存逐样本全网络SOP，用于计算预算差异的配对统计；如果已有监控器无法直接提供，则仅在正确性验证通过后增加诊断功能。

不要为获得逐样本SOP而改变模型正常前向语义。

所有SOP按实际样本数量加权。

官方测试集始终隔离。

## 13. 配对统计

固定Bootstrap重采样次数2000。

seed=20261128。

记录实际随机数生成器及区间计算方法。

按相同样本ID和标签严格配对。

主要比较：

G1-G0。

G1-G2。

次要比较：

G1-B0。

G0-B0。

G2-B0。

报告：

- 准确率百分点差。
- 配对CE差。
- 95%配对Bootstrap置信区间。
- 参考组错误、实验组正确的数量。
- 参考组正确、实验组错误的数量。
- 两组预测类别不同的数量。
- 实际SOP差异。

如果获得逐样本SOP，报告相应配对成本差及置信区间。

样本级置信区间不能代替跨训练种子的稳定性分析。

不宣称已经完成独立统计确认或正式非劣效性检验。

## 14. 推进门槛

预注册判定条件：

1. 父模型、数据、公式、状态及结果重建检查通过。
2. G1固定q=0.04。
3. G0/G2在D_cal与G1降幅差分别不超过0.1个百分点。
4. 三组校准降幅均在4%—6%。
5. G1验证集SOP降幅位于4%—6%。
6. G0/G2在验证集上的实际SOP降幅与G1分别相差不超过1个百分点。
7. G1相对B0准确率下降不超过0.5个百分点。
8. G1相对G0准确率优势至少0.3个百分点。
9. G1相对G2准确率优势至少0.3个百分点。
10. 权重掩码、阈值、BN运行统计及非目标参数保持不变。

第8、9项延续EXP-020原性能推进门槛，不放宽。

置信区间用于证据强度判断，不将探索性点估计门槛通过自动解释为统计显著。

如果全部通过：

建议下一轮进入正式短程训练干预，由理论研究员进一步审查。

如果成本匹配通过但G1性能没有优势：

明确报告当前局部方法的计算效率优势未能转化为任务性能优势。

不得仅因为H-019已获得支持就自动认为H-017成立。

如果G1与G0/G2准确率相近，但CE或参数更新规模存在差异：

作为次要探索性观察保留，不能替代原主要门槛。

如果实际验证预算不匹配：

H-017仍缺乏公平比较证据，不扩大搜索。

## 15. 理论分析边界

MP-EFR通过有效连接数量调整通道活动惩罚。

它没有显式建模任务损失敏感性。

更强的局部计算成本下降可能伴随更大任务性能损失。

因此本轮重点是精度—SOP权衡，而不是再次证明G1能够更快减少SOP。

不增加任务梯度保护、蒸馏、注意力、EMA、动态rho或其他机制。

不将R81/R82描述为学术首创。

本轮没有新增科学算法，只补全缺失的公平对照。

## 16. 资源预算

需要用户新授权。

建议：

- 单张符合实验室共享管理规定的RTX4090。
- GPU总占用上限15分钟。
- CPU累计计算上限30分钟。
- 正式训练Epoch=0。
- 每组有限预算匹配候选数量按第10节限制。

EXP-020历史累计保守墙钟约88.67秒。本轮预计计算规模较小，但仍须以真实硬件和共享负载为准。

不得干扰其他GPU用户。

所有检查、重试、局部反向及评价计入预算。

若发生：

- 父检查点校验失败。
- 原实验结果无法合理重建。
- 数据隔离失败。
- 有效权重掩码发生变化。
- BN运行统计发生改变。
- 非有限参数或损失。
- 超预算。
- 无法保持公平实验条件。

停止受影响任务并报告。

不得自动增加rho候选、目标层、随机置换数或训练轮数。

## 17. 科研状态更新

追加：

research/ROUND-019-TASK.md

更新：

- THEORY.md，追加v0.2.5及R81—R82。
- HYPOTHESES.md，按理论研究员判定更新H-019和H-017。
- RESEARCH_STATE.md。
- EXPERIMENT_PLAN.md。
- EXPERIMENT_REGISTRY.csv。
- DECISION_LOG.md，追加DEC-019。

保留EXP-020为COMPLETED且原推进门槛失败。

新增EXP-021记录。

不能将EXP-021新协议反向用于宣称EXP-020已经通过。

最终科学结论仍由理论研究员审查。

## 18. 实验交付

建立独立experiments/EXP-021及唯一Run ID。

至少保存：

- TASK.md
- config.json
- environment.txt
- code_version.json
- commands.sh
- status.json
- parent_checkpoint_audit.json
- historical_reconstruction_audit.json
- data_split_audit.json
- gradient_vectors.csv
- gradient_statistics.csv
- formula_tests.json
- coefficient_audit.json
- calibration_response.csv
- budget_matching_trace.csv
- selection_lock.json
- parameter_integrity_audit.json
- update_norm_audit.csv
- validation_summary.csv
- validation_per_sample.csv
- layer_sops.csv
- paired_bootstrap.json
- hypothesis_gate_results.json
- resource_usage.csv
- REPORT.md
- limitations.md
- run.log
- figures/

建议绘制：

1. 有限预算匹配轨迹。
2. 三组实际计算预算匹配情况。
3. 匹配预算下的验证准确率及CE。
4. 各组配对准确率差及置信区间。
5. 实际参数更新范数与SOP收益。

未执行项目标记NA并说明原因。

保留所有负面数据和失败记录。

## 19. GitHub最终交付

从上轮最终结果提交建立独立实验分支。

正式运行前提交任务、固定配置、判定门槛及执行代码。

完成校准匹配后，在验证前提交实际selection lock。

实验结束后同步全部真实结果和科研状态。

不修改main，不force push，不覆盖历史实验。

最终返回：

- GitHub实验分支链接。
- 实际执行代码SHA。
- 最终结果提交SHA。
- EXP-021执行状态。
- EXP-020历史结果重建情况。
- G0/G1/G2实际锁定q。
- 实际校准预算匹配误差。
- 各组真实验证Accuracy、CE、FR、SOP及稀疏率。
- 配对准确率差及Bootstrap置信区间。
- 预注册推进门槛结果。
- 全部异常及限制。
- 实际CPU/GPU消耗。
- 实验报告和原始数据路径。

本轮只完成MP-EFR公平预算对照，不进行正式训练。

获得用户授权后执行。