# Codex科研执行任务：ROUND-024

# UniSparse高比例结构化剪枝与固定支持集分类读出重构

## 1. 任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-024
- 实验编号：EXP-026
- 科研决策：DEC-024，HOLD BASELINE SWITCH + DIAGNOSE
- 理论版本：v0.3.0
- 科学问题：Q-014，高压缩率下的结构冗余与任务输出可恢复性
- 新科研假设：H-026、H-027
- 项目仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-023-structured-baseline-audit
- 本轮建议分支：research/round-024-structural-reconstruction
- 上轮最终结果SHA：64f274a7520e24662e86f0a811560c5af1011d71
- EXP-024最终结果SHA：7d5bdba3532ccb5b7dc3a8d69053c39314fe4a93
- EXP-024实际执行代码SHA：c494b5876de2dc1e0beb62c32137a0be63f4d19a
- UniSparse官方固定SHA：b5b3da4fcd2e7abcd2c20c93d41cf63ef3201daa
- 依赖实验：EXP-018、EXP-019、EXP-024、EXP-025

P0：归档候选Baseline静态准入结果。

P0：继承EXP-024结构化剪枝及严格FP32评价协议。

P1：验证高压缩率下简单剪枝方法的性能。

P1：验证固定连接支持集线性读出重构的实际恢复能力。

不切换Baseline，不开展SGD训练，不恢复此前已停止的候选方法。

## 2. ROUND-023科研判定

EXP-025为COMPLETED。

C-DPSS、SCA、Dynamic Spatio-Temporal Pruning均保持UNDER_REVIEW。

原静态评分：

C-DPSS：6.10。

SCA：5.00。

Dynamic ST：4.15。

这些评分不代表实际训练表现。

C-DPSS主要阻塞：

- 代码许可证未确认。
- 论文全文未确认。
- 预训练权重未确认。
- 官方CIFAR加载路径使用测试集作为验证集进行模型选择。

SCA官方代码及预训练权重仍未确认。

暂缓正式切换Baseline。

UniSparse继续作为当前Controlled Baseline。

EXP-025 CPU累计计量缺失必须保留。

## 3. 新增文献记录

更新LITERATURE.md。

Spik4lite: Refactoring Neuromorphic Sparsity for Efficient Spiking Neural Networks on Commodity Edge Devices。

ICML 2026。

会议主页：

https://icml.cc/virtual/2026/poster/63522

代码：

https://github.com/yongzhishe/Spik4lite

理论研究员已核验固定SHA：

4d33fca2576680eb114744340df51e926aad2f2f

GitHub许可证显示Apache-2.0。

仓库自称演示实现，包含SNN Transformer训练和推理入口。

实际环境兼容性、权重及完整复现能力仍未确认。

仅添加候选及文献记录，不导入源码、不训练。

Optimal Spiking Brain Compression：

https://arxiv.org/abs/2506.03996

He, Zhang, Sun, Channel Pruning for Accelerating Very Deep Neural Networks, ICCV 2017：

https://openaccess.thecvf.com/content_iccv_2017/html/He_Channel_Pruning_for_ICCV_2017_paper.html

最小二乘重构及剪枝后恢复已有相关研究。

本轮不得将线性重构或闭式解描述为学术首创。

## 4. EXP-024真实依据

原严格FP32模型：

Accuracy=94.222222%。

CE=0.1727328626844618。

SOP=86009551.08977778。

参数量36720484。

25%隐层剪枝组参数量28280164。

L1组S1：

Accuracy=94.333333%。

CE=0.17019318421681723。

SOP=82181961.79888889。

随机组S0：

Accuracy=94.111111%。

FR组S2：

Accuracy=93.733333%。

M5：

Accuracy=93.977778%。

M5未通过原性能推进门槛，继续保持STOPPED。

本轮不重新评价M5，不搜索block_size。

EXP-024真实结果提示最后隐层可能具有大量结构冗余，但尚未测定50%及75%剪枝率。

## 5. 固定父模型

使用：

experiments/EXP-018/runs/EXP-018-CONTINUE-001/U2/epoch-300.pth

SHA256：

19b1312cbaae49ad95a7d001270c4357d7ea23b2584d19698b0084b3da4b094f

保持只读。

运行前后核对哈希。

不使用其他best_validation或中间Epoch检查点。

复用EXP-024实际验证通过的模型加载及紧凑结构构建实现。

先检查仓库确定真实代码路径，不虚构函数名称。

不修改vendor/UniSparse源码。

## 6. 固定环境与精度

沿用EXP-024 RUN-002的真实配置。

CIFAR-10。

CifarNet。

base_channels=256。

T=4。

官方CuPy LIF。

FP32。

关闭cuDNN TF32。

关闭matmul TF32。

关闭AMP。

Batch Size=64。

workers=4。

model.eval()。

确定性评价预处理。

每批前后按照原协议正确复位神经元状态。

不混用EXP-020的默认后端B0结果。

本轮首先重建严格FP32 B0。

若不能合理重建EXP-024的原始结果，停止正式比较并报告。

## 7. 数据隔离

沿用EXP-020及EXP-024的实际样本索引。

D_grad：

512张，用于剪枝集合选择及读出重构拟合。

D_cal：

512张，仅用于拟合后输出重构泛化诊断。

固定验证集：

4500张，用于最终评价。

完整划分文件：

experiments/EXP-004/splits/indices.json

SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

D_grad索引哈希：

93c4c834c3fd3c8b92a64caa88da8a671e27be23dc2df11999c8e5f3dfbe47d8

D_cal索引哈希：

b671cbd803b13cc76b02f5db4bad311ef9492276622051df01af56a931301baa

核查两组诊断样本实际索引及互斥性。

D_grad和D_cal均来自父模型训练集，不能称为模型未见样本。

D_cal相对于本轮重构拟合必须独立。

4500张验证样本已经多轮用于开发。

不得根据D_cal或验证集改变重构正则系数、剪枝集合及剪枝率。

不读取官方test_batch。

不使用旧500张诊断池。

## 8. 固定结构剪枝预算

最后隐层H=2048。

主要条件：删除50%，即1024个神经元。

辅助探索：删除75%，即1536个神经元。

两个比例均提前固定。

不搜索其他比例，不根据验证结果选择有利比例替代主要条件。

理论结构参数：

每删除一个神经元，删除16485个参数。

R114：

P(r)=36720484-2048*r*16485

R115：

R_P(r)=(36720484-P(r))/36720484

50%隐层剪枝预计剩余19839844个参数。

75%隐层剪枝预计剩余11399524个参数。

实际物理参数量必须独立统计。

## 9. 实验分组

B0：

原始未剪枝模型。

R0：

随机剪枝。

固定三个选择种子：

20261128、20261129、20261130。

采用numpy.default_rng PCG64。

按实际神经元索引无放回生成固定排列。

两个剪枝比例使用同一排列的对应前缀。

报告全部三个种子，不选择最好结果。

R1：

fc1有效权重L1剪枝。

score[i]=sum_j(abs(W1_eff[i,j]))。

删除得分最小者。

R2：

最后隐层时间平均脉冲发放率剪枝。

score[i]=mean_n(r[n,i])。

D_grad用于活动估计。

R3：

与R1完全相同的保留神经元集合。

在R1真实紧凑模型基础上执行固定支持集分类读出重构。

不改变剪枝集合。

所有排序平局按较小原始索引优先。

两个剪枝比例均执行R0三个随机集合、R1、R2及R3。

B0只需统一评价一次。

## 10. R116：删除贡献计算

复用EXP-024的R101—R108。

捕获：

activation_fc1真实二值脉冲。

s.shape=[T,B,H,1,1]。

时间平均：

R[n,i]=mean_t(s[t,n,i])。

R.shape=[N,H]。

有效fc2权重：

W2_eff=parent.fc2.sparse_weight()。

W2_eff.shape=[100,H]。

类别读出：

A[k,i]=mean(W2_eff[10*k:10*k+10,i])。

b[k]=mean(fc2.bias[10*k:10*k+10])。

原输出：

Z=R@A.T+b。

设保留集合J，删除集合S。

R116：

E=R[:,S]@A[:,S].T。

E.shape=[N,10]。

必须与实际紧凑模型删除前后分类输出差一致。

使用FP64进行离线代数计算，真实完整网络前向采用严格FP32。

## 11. R117—R119固定支持集重构

对于原紧凑模型fc2有效掩码M：

M.shape=[100,len(J)]。

每类对应10个输出单元。

m[k,i]=sum(M[10*k:10*k+10,i])。

允许补偿的支持集：

I_k={i:m[k,i]>0}。

禁止在原来全部无效的类别-神经元连接上新增补偿。

对于每个类别：

X=R[:,J][:,I_k]。

e=E[:,k]。

R118：

min_delta ||X@delta-e||²/N+lambda_k*||delta||²。

本轮固定正则尺度：

lambda_k=0.001*mean_j(sum_n(X[n,j]²)/N)。

该尺度为本轮预设工程选择，不宣称理论最优。

不搜索正则系数。

若I_k为空或lambda_k=0，则该类补偿为0并记录退化。

不得为获得非零补偿而临时修改公式。

R119：

delta=solve(X.T@X+N*lambda_k*I, X.T@e)。

当使用对偶形式更低开销时：

delta=X.T@solve(X@X.T+N*lambda_k*I,e)。

禁止显式求逆。

使用FP64稳定线性方程求解。

记录残差、lambda、矩阵维度和非有限检查结果。

保存补偿向量及哈希。

## 12. R120映射回实际fc2权重

对于I_k中的神经元i：

仅允许在原M[j,i]=1的连接上执行：

delta_W_eff[j,i]=10*delta[k,i]/m[k,i]。

其他原无效连接保持不变。

修改后的目标有效权重：

W_eff_new=W_eff_old+delta_W_eff。

只修改紧凑模型的fc2。

fc1、LIF、BN及其他全部参数保持不变。

保留原fc2 bias和稀疏阈值。

若官方权重映射为：

W_eff=sign(W_raw)*relu(abs(W_raw)-d)

且d为非负有限阈值，则非零目标有效权重可通过：

W_raw_new=sign(W_eff_new)*(abs(W_eff_new)+d)

映射回潜在权重。

目标有效权重恰好为0时，保证其实际有效权重为0。

原无效连接的潜在权重保持原值。

所有映射必须依据实际官方实现核查。

若存在其他权重参数化、阈值或数值语义，暂停相关实现并报告，不擅自修改公式。

映射后重新计算实际有效掩码。

必须满足新有效掩码是原有效掩码的子集。

允许原有效连接因补偿恰好变为零，但不得激活原无效连接。

必须重新核对真实读出权重及完整网络logits。

不能只报告离线计算结果。

## 13. 最低正确性检查

运行正式实验前至少完成：

1. 复用EXP-024全部必要结构正确性检查。
2. 严格FP32真实输出与FP64读出重构。
3. 真正2+1非等尾批检查。
4. R116与实际删减logits差一致。
5. 人工小矩阵的原始及对偶ridge解一致。
6. 有效连接支持集为空的处理。
7. 全零输入的退化处理。
8. 非有限数值检测。
9. R120经过Boost后的分类权重补偿正确。
10. 原软阈值权重映射的正负值、边界及零值检查。
11. 补偿后无新增有效连接。
12. 非目标参数及BN运行统计完全不变。
13. R1/R3的保留神经元脉冲完全一致。
14. 两组物理参数预算完全一致。
15. 捕获和关闭捕获的正常前向一致性。

归一化logits误差容差沿用1e-5。

不能放宽容差掩盖错误。

每项检查保存实际输入、断言及真实退出状态。

未执行测试不得标记通过。

## 14. 执行及选择锁

先完成全部预定比例及组别的神经元选择。

使用D_grad拟合R3分类读出补偿。

保存：

- 所有随机排列。
- L1及FR完整得分。
- 每组保留和删除索引。
- 每个类别固定支持集。
- lambda_k。
- 补偿向量。
- 补偿前后有效权重哈希。
- 原父检查点及样本哈希。
- 固定判定条件。

在任何D_cal或4500张验证集评价前，将selection_lock提交GitHub。

锁定后不得改变剪枝集合、lambda或补偿公式。

本轮不使用验证集调参。

## 15. 评价指标

对于B0及全部紧凑模型记录：

Accuracy。

CE。

实际物理参数量。

真实结构压缩率。

纯推理模型文件大小。

有效非零权重数。

有效权重稀疏率。

全网络FR。

SOP_sample_weighted。

SOP_author。

fc1、fc2及逐层SOP。

逐样本预测、标签、正确性及CE。

对于R1/R3，额外记录D_grad和D_cal上的：

教师logits均方重构误差。

RMSE。

教师与紧凑模型预测一致率。

实际分类CE。

补偿前后有效权重L2距离。

补偿前后有效连接支持集变化。

线性系统求解耗时。

D_grad和D_cal必须分别报告。

拟合集重构误差下降不能替代验证集性能恢复。

## 16. 科研假设与推进门槛

H-026主要条件：

在固定50%隐层剪枝下，R1相对B0准确率下降至少1个百分点。

75%为辅助探索，不反向替代50%主要假设判定。

H-027主要条件：

固定50%隐层剪枝下：

- R3在D_cal的教师logits均方误差低于R1。
- R3相对R1验证准确率提高至少0.5个百分点。
- R3验证CE低于R1。
- R3与R1物理参数量相同。
- R3有效连接支持集不扩大。
- R3实际全网络SOP不超过R1的101%。
- 所有正确性及数据隔离检查通过。

以上是探索性推进标准，不是统计显著性的定义。

75%结果按相同指标独立报告。

如果H-026未获得支持：

说明主要比例尚未观察到预设幅度的简单剪枝性能损失。

如果H-026获得支持但H-027失败：

说明当前固定读出补偿未能达到预设的任务恢复要求。

如果H-027通过：

仅报告该重构机制的可行性。

不将其直接作为具有独立原创性的最终方法。

所有正式比较完成后，不自动增加剪枝率、调整lambda或开展SGD训练。

## 17. 配对统计

固定2000次配对Bootstrap。

seed=20261128。

numpy.default_rng PCG64。

百分位95%区间。

主要比较：

50% R1-B0。

50% R3-R1。

辅助比较：

75% R1-B0。

75% R3-R1。

以及各比例R3相对R2和全部随机集合的真实表现。

报告Accuracy百分点差、CE差和配对区间。

随机集合报告均值及标准差，但三个随机选择种子不能替代三个独立训练种子。

未采集逐样本SOP时，其配对区间标记NA。

不得宣称独立统计确认。

## 18. 资源预算

需要用户新授权。

建议：

单张符合实验室管理规定的RTX4090。

GPU总占用上限20分钟。

CPU累计计算上限45分钟。

SGD训练Epoch=0。

新增GPU操作仅限：

必要正确性测试、脉冲捕获、紧凑模型构建、线性求解和固定评价。

允许CPU进行线性系统求解。

所有预检、失败重试和正式运行均计入预算。

CPU资源计数必须从本轮审查开始尽可能覆盖整个进程树。

不能再次仅用报告生成脚本时间代表全程消耗。

不干扰其他用户进程。

不修改原系统环境和驱动。

## 19. 停止条件

若发生：

- 父检查点哈希不一致。
- 严格FP32基线重建失败。
- 真实结构删除不成立。
- R116/R120公式与真实前向不一致。
- 有效连接支持集扩大。
- 非目标参数变化。
- 线性系统非有限。
- 数据隔离失败。
- 必要准入测试未完成。
- OOM。
- 资源超限。

停止相应任务并如实报告。

不因异常调整研究目标或降低正确性标准。

保留全部失败记录。

## 20. 科研状态文件更新

新增：

research/ROUND-024-TASK.md

更新：

- THEORY.md：追加v0.3.0及R113—R120。
- HYPOTHESES.md：新增H-026、H-027。
- RESEARCH_STATE.md。
- EXPERIMENT_PLAN.md。
- EXPERIMENT_REGISTRY.csv。
- DECISION_LOG.md：追加DEC-024。
- LITERATURE.md。
- STRUCTURED_BASELINE_COMPARISON.md。

EXP-025保持COMPLETED。

三项原候选Baseline保持UNDER_REVIEW。

Spik4lite添加为新文献及备选审查对象，不标记ENGINEERING_ADMITTED。

保留EXP-024原门槛失败、M5停止状态及所有历史负面结果。

## 21. 实验交付

建立独立：

experiments/ROUND-024/

experiments/EXP-026/

至少保存：

- TASK.md
- config.json
- environment.txt
- code_version.json
- commands.sh
- status.json
- parent_checkpoint_audit.json
- precision_audit.json
- data_split_audit.json
- baseline_reconstruction_audit.json
- admission_gate_audit.json
- formula_tests.json
- neuron_statistics.csv
- selected_neurons.json
- reconstruction_coefficients.csv
- readout_support_audit.json
- structural_equivalence_tests.json
- selection_lock.json
- reconstruction_summary.csv
- validation_summary.csv
- validation_per_sample.csv
- layer_sops.csv
- parameter_count_audit.json
- paired_bootstrap.json
- hypothesis_gate_results.json
- resource_usage.csv
- checkpoint_manifest.json
- REPORT.md
- limitations.md
- run.log
- figures/

绘制：

1. 结构压缩率与Accuracy的关系。
2. 高压缩率下随机、L1及FR性能。
3. R3补偿前后D_grad/D_cal重构误差。
4. 相同参数预算下R1与R3性能。
5. 配对准确率差及置信区间。
6. 理论SOP与实际物理参数量。

保存所有真实负面结果。

未执行文件或指标标记NA及原因。

大型模型权重保留服务器，记录实际路径、结构配置和SHA256。

## 22. GitHub交付

从ROUND-023最终结果提交创建独立分支。

遵守AGENTS.md。

正式实验前提交任务、代码及固定门槛。

评价前提交选择和补偿锁。

结束后提交全部真实结果。

不修改main，不force push，不覆盖历史记录。

最终返回：

- GitHub实验分支链接。
- 实际执行代码SHA。
- 选择锁SHA。
- 最终结果提交SHA。
- EXP-026执行状态。
- B0历史结果重建情况。
- 50%与75%各组Accuracy、CE、FR、SOP及实际参数量。
- 三个随机集合全部结果。
- R1与R3重构误差。
- 有效连接支持集审计。
- 配对统计及推进门槛结果。
- 真实CPU/GPU消耗。
- 全部异常及实验局限。
- 实验报告及原始数据路径。

本轮是一项高压缩率研究空间与损失可恢复性验证。

不擅自切换Baseline，不宣称最小二乘重构为学术创新，不启动额外训练。

获得用户资源授权后执行。