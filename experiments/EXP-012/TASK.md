# Codex 科研执行任务：ROUND-012

# AT-LIF 共享阈值梯度尺度修正及 ARTG 有效干预验证

## 一、任务身份

- 项目：SNN-AP Research
- 当前轮次：ROUND-012
- 科研阶段：阶段3—4，实验干预修正及候选方法重新验证
- 科研决策：DEC-012，REPAIR + REPLICATE
- 理论版本：v0.1.8
- 科学问题：Q-008，AT-LIF阈值代理梯度优化
- 新科研假设：H-012、H-013
- 实验编号：EXP-012、条件性EXP-013
- GitHub仓库：https://github.com/corex197-pixel/SNN-AP-Research
- 上轮分支：research/round-011-threshold-gradient
- 本轮建议分支：research/round-012-effective-threshold-gradient
- 上轮修复后执行代码SHA：10a52e75c5cde3ebb972e6b58130f42bf7e5b04b
- 上轮结果SHA：1fdace017f755a1504e55ee0ec493c6502ae00dd
- 依赖实验：EXP-004、EXP-007、EXP-011
- 优先级：P0有效阈值干预检查；P1条件性正式训练；P2同预算评价

首先读取AGENTS.md、THEORY.md、HYPOTHESES.md、RESEARCH_STATE.md、ROUND-011-TASK.md、EXP-011的报告、原始梯度及更新数据。

从上轮结果SHA建立独立实验分支。

## 二、ROUND-011正式科研结论

EXP-011完成。

G0、G1、M4三组最终模型参数及buffers逐张量完全一致。

在20%预算条件下：

- C0：Accuracy=90.8%，SOPs=43652928.192。
- G0/G1/M4：Accuracy=89.8%，SOPs=42915156.551。
- M4相对C0准确率差=-1.0个百分点。
- 配对95% CI=[-1.7778,-0.2]个百分点。

在30%预算条件下：

- C0：Accuracy=89.6222%。
- G0/G1/M4：Accuracy=89.4444%。

原H-011实用性筛选失败。

H-011正式判定为INSUFFICIENT_EVIDENCE，原协议下没有性能收益，且梯度规则未产生可分辨的最终模型差异。

不得将此扩大为不同代理梯度机制普遍无效。

保留ROUND-011全部前置失败、混合精度修复、训练中断和恢复记录。

## 三、理论研究员新增证据

读取：

experiments/EXP-011/threshold_updates.csv

实际共有285条层级诊断记录，每组95条。

其中：

- 285条任务阈值梯度非零。
- 285条实际optimizer阈值改变量均为0。
- 282条手动阈值更新非零。

任务梯度绝对值中位数：

- G0：9.21578014e-9。
- G1：1.13769429e-8。
- M4：7.90327181e-9。

使用各条记录的实际学习率乘以任务梯度绝对值，得到名义SGD更新量中位数：

- G0：2.2580915984410954e-11。
- G1：3.006815063357364e-11。
- M4：2.0732356291708835e-11。

上述名义更新未包含Momentum及Nesterov的实际作用。

已记录的实际optimizer改变量为0。

手动阈值实际更新量绝对值中位数约1.14440918e-5。

阈值数值范围约12.63至66.82。

当前最有依据的解释是：均值聚合导致的任务梯度尺度过小，使任务阈值更新无法有效作用于FP32参数。

现有记录只覆盖每轮指定诊断Batch，不能断言全部训练步骤的optimizer更新均为0。

## 四、理论公式

保留R1—R39。

新增R40—R46。

### R40：共享阈值梯度聚合

设每个时间步共享阈值涉及N个元素，局部代理梯度项为v_i。

G_mean=mean_i(v_i)

G_sum=sum_i(v_i)

在精确算术下：

G_sum=N*G_mean

注意mean本身是一种优化缩放设计，不自动等于实现错误。

### R41：FP32数值分辨率

对正常正FP32数值theta，其所在二进制区间的典型ULP：

ULP(theta)=2**(floor(log2(theta))-23)

实际程序中优先使用torch.nextafter计算上下相邻可表示数值。

记录：

ulp_up=nextafter(theta,+inf)-theta

ulp_down=theta-nextafter(theta,-inf)

需要处理二进制区间边界处上下间隔不同的情况。

低于相应舍入边界的更新可能无法改变参数实际存储值。

### R42：更新退化现象

theta_next≈theta_current+manual_pruning_delta

仅当optimizer任务更新无法实际改变阈值时，该近似成立。

不得将其作为所有训练过程的普遍结论。

### R43—R45

定义：

r=m/theta

g=max(1-abs(r-1),0)

G0-S：

grad_theta=-sum_i(q_i*g_i*g_i)

G1-S：

grad_theta=-sum_i(q_i*g_i)

M4-S：

grad_theta=sum_i(a_i*(s_i-r_i*g_i)-b_i*r_i*g_i/theta)

其中q、a、b及二值脉冲定义沿用ROUND-011的R32—R39。

三组保持各自原代理梯度因子不变，仅统一改变聚合方式。

### R46：等价的梯度后缩放

在每次共享阈值调用所涉及的张量元素数量相同的条件下：

theta.grad <- N*theta.grad

N为实际单时间步元素数量。

对于[B,C,H,W]：

N=B_actual*C*H*W

各时间步的梯度已由PyTorch累加。

禁止再次乘T。

禁止重复放大已经采用sum聚合的梯度。

## 五、研究目标与科研假设

H-012，UNTESTED：

原均值聚合严重抑制当前模型的任务阈值实际更新；恢复求和聚合能够产生可分辨的优化器阈值更新。

H-013，UNTESTED：

在恢复有效任务阈值更新后，相同聚合尺度下的ARTG能够获得优于原代码代理因子及论文对齐因子的精度—SOPs权衡。

H-012成立不意味着H-013成立。

优化器更新非零只是干预有效性的必要检查，不代表分类性能一定改善。

## 六、代码修改及数值实现

优先复用修复后的：

experiments/ROUND-011/artg.py

experiments/ROUND-011/run.py

保留G0/G1/M4的原始自定义反向函数。

新增独立的可选阈值梯度聚合开关：

- mean
- sum_equivalent

sum_equivalent建议在scaler.unscale_(optimizer)之后，scaler.step(optimizer)之前执行。

实际步骤：

1. 正常前向。
2. 计算原交叉熵。
3. scaler.scale(loss).backward()。
4. scaler.unscale_(optimizer)。
5. 检查当前原始梯度有限性。
6. 仅将ATLIF共享阈值grad乘以该模块实际单时间步元素数N。
7. 再次检查全部被放大的梯度有限性。
8. scaler.step(optimizer)。
9. 按原策略执行threshold_update(model,lr)。
10. scaler.update()。
11. 检查阈值有限且严格为正。
12. 记录实际参数更新。

如果反缩放时已经存在非有限梯度，按原AMP跳步策略处理，不强行执行优化器。

如果梯度后缩放引入新的非有限值，必须停止受影响运行并保留日志，不能依赖此前已经完成的AMP溢出检查。

禁止自动裁剪梯度、修改公式或调整梯度倍率。

确保：

- 只放大阈值任务梯度。
- 不放大手动稀疏更新量。
- 不改变其他参数梯度。
- 不改变原优化器动量、Nesterov、参数组和weight decay设置。
- 核查阈值参数实际是否应用weight decay，单独记录。
- 保存实际N，而不是硬编码推断的网络尺寸。

如果同一模块共享阈值存在不同大小张量调用，不能简单乘单一N。必须报告并采用经理论确认的逐调用聚合方式。

禁止修改原Baseline核心文件。

## 七、固定模型与数据

原始第194轮最佳检查点：

experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth

SHA256：

c068f54379cb5d29bcfa5fe2d9de676cd3cc153433ada1ed5aa1c66355d0b32a

固定数据划分：

experiments/EXP-004/splits/indices.json

SHA256：

69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165

固定500张训练来源校准集：

experiments/EXP-007/calibration_indices.json

SHA256：

b849b74ef19cb4a09df756c54871122f3b02aa047c704d0436feab15a38bde52

使用原45000张训练样本及4500张验证样本。

所有模型均从原194轮检查点独立初始化。

19个ATLIF阈值初始乘以1.25。

禁止使用ROUND-011已经训练的最终模型继续训练。

禁止使用官方测试集或已经消耗的500张科研诊断池。

## 八、正式训练前的正确性测试

必须全部在任何正式训练之前完成。

### 8.1 人工聚合测试

对于G0、G1、M4：

- 检查原mean梯度与R32、R33、R39一致。
- 检查独立sum参考实现与R43—R45一致。
- 检查N乘mean的结果与sum参考一致。
- 检查多时间步梯度累加。
- 检查不同张量维度、不同Batch Size。
- 检查最后一个不完整Batch。
- 检查共享标量阈值形状。
- 检查有符号梯度相消。
- 检查单独幅值、重置及双路径梯度。

float64归一化误差容差1e-10，float32容差1e-5。

同时记录实际绝对误差，不能仅依赖归一化误差掩盖微小梯度不一致。

### 8.2 混合精度测试

继承ROUND-011所有已经修复的混合精度测试，包括大上游梯度、float16膜电位、float32阈值以及三条梯度路径。

必须验证：

- AMP反缩放之后的梯度尺度正确。
- 放大后所有阈值梯度保持有限。
- 成功optimizer更新与AMP跳步能够区分。
- 后缩放引入非有限值时能够正确停止。
- 不影响非阈值参数更新。

### 8.3 真实网络单步检查

使用相同原始检查点、相同输入、相同参数和上游损失。

在FP32及正式AMP协议下分别检查：

- 三种代理梯度的前向输出。
- 完整二值脉冲。
- 原始SOPMonitor计数。
- 非阈值参数梯度。
- 各层实际N。
- mean任务梯度。
- sum等价任务梯度。
- 实际Momentum buffer。
- 学习率。
- FP32上下ULP。
- optimizer之前和之后的阈值。
- 手动更新前后的阈值。

必须至少观察一次成功的实际优化器更新，不能将AMP跳步后的零变化当作数值分辨率证据。

## 九、EXP-012：短程有效干预诊断

实验组：

D0：G0原始mean聚合。

G0-S：原平方因子sum等价聚合。

G1-S：论文对齐因子sum等价聚合。

M4-S：ARTG sum等价聚合。

所有组从相同P0初始化独立开始。

使用相同固定训练数据流，seed=2030。

相同SGD、Momentum、Nesterov、学习率0.005、原手动阈值更新机制。

每组最多16个有效优化器更新，最多20次Batch尝试。

不运行验证集，不根据准确率决定是否调整公式。

详细记录每个训练步全部19个阈值的：

- mean task gradient。
- actual scaling factor N。
- scaled task gradient。
- learning rate。
- optimizer momentum buffer。
- FP32 ULP。
- nominal SGD update。
- optimizer actual delta。
- manual actual delta。
- final threshold。
- firing rate。
- AMP skip。
- nonfinite values。

对首次成功更新进行独立优化器数值检查。

比较任务更新被舍入的情况，而不是只比较梯度张量是否非零。

### H-012进入门槛

以下条件必须全部满足：

1. G0-S、G1-S、M4-S均完成16个有效优化器更新。
2. 三个sum组的阈值、损失及梯度保持有限，阈值严格为正。
3. 每个sum组至少10/19个模块在短程实验中出现实际非零optimizer阈值更新。
4. M4-S与G0-S、G1-S的最终阈值向量分别存在至少5个模块的绝对差异超过8个局部ULP。
5. 数据输入、初始化及必要工程正确性检查通过。
6. 不存在明显的更新爆炸或协议失效。

对于两组阈值比较，使用两者局部ULP中的较大者。

如AMP跳步位置不同，必须记录数据消费与更新步数差异；不能把这种短程轨迹差异单独归因于代理梯度。

上述阈值属于工程干预有效性验收，不代表已证明科学性能收益。

若门槛未通过：

- EXP-013标记BLOCKED。
- 保存全部实际数据。
- 不增加梯度倍率。
- 不修改学习率。
- 不擅自切换至Adam。
- 不重新设计梯度公式。
- 直接结束本轮并报告原因。

## 十、条件性EXP-013：正式性能先导实验

仅在EXP-012全部门槛通过后执行。

实验组：

G0-S、G1-S、M4-S。

每组重新从相同原194轮检查点初始化，阈值乘以1.25。

不能使用EXP-012短程诊断后的模型及优化器状态。

固定配置：

- CIFAR-10。
- ResNet-20。
- T=8。
- tau=0.9。
- sp=0.002。
- eta2=0。
- Batch Size=128。
- num_workers=4。
- Epoch=5，每组。
- seed=2030。
- SGD。
- momentum=0.9。
- nesterov=True。
- initial_lr=0.005。
- weight_decay=5e-4。
- CosineAnnealingLR。
- T_max=5。
- eta_min=0.0001。
- AMP=True。
- TET=False。
- label_smoothing=0。

沿用ROUND-011的参数组、训练增强、训练顺序和AMP手动更新策略。

不加入KD、SOPs惩罚、BN校准或其他机制。

各组最终主结果固定为第5轮检查点。

禁止根据验证表现选择最有利的Epoch。

每轮保存检查点并记录真实运行代码SHA。

## 十一、训练全过程干预检查

本轮必须比ROUND-011更完整地记录实际阈值更新。

允许对全部训练Batch的19个标量阈值采用轻量统计，并按Epoch汇总：

- optimizer更新为零的比例。
- 各层optimizer实际更新绝对值累计量。
- 各层manual实际更新绝对值累计量。
- 任务与手动更新量的比例。
- 任务更新的正负比例。
- 相对于FP32 ULP的更新幅度。
- 任务梯度及手动梯度竞争情况。
- AMP成功和跳步次数。

额外在每轮固定第9个Batch保存逐层详细记录。

M4-S保留幅值项、发放项、重置项的诊断。

区分缩放前和缩放后的梯度组件。

不保存全部膜电位张量。

检查最终模型状态是否真正产生差异。

若三组最终仍逐张量完全一致，不得宣称H-013获得有效验证。

## 十二、同SOPs预算正式评价

沿用ROUND-011评价协议。

先报告第5轮最终模型未经校准的直接性能。

然后保持各组学习得到的相对阈值结构：

theta_eval[l]=alpha*theta_learned[l]

固定校准倍率：

[0.75,0.8,0.9,1.0,1.1,1.2,1.25,1.3,1.4,1.5,1.75,2.0]

20%目标为主要目标。

30%目标为辅助目标。

固定500张校准图片，按实际SOPs选择配置。

优先选择校准降幅位于[target,target+0.02]的候选。

如需要，按ROUND-011规则最多3次中点细化。

不根据准确率选择倍率。

所有配置锁定之后评价原4500张验证图片。

C0仍为原始权重和原阈值向量的简单统一阈值校准。

使用原作者SOPMonitor及完全相同的统计范围。

不将SOPs描述为硬件实测能耗。

## 十三、H-013预设判定标准

主要20%预算下：

1. M4-S验证SOPs降幅至少20%。
2. M4-S相对原B0准确率损失不超过1个百分点。
3. M4-S与G0-S、G1-S、C0的实际SOPs降幅差分别不超过2个百分点。
4. M4-S准确率分别比G0-S、G1-S、C0至少提高0.5个百分点。
5. 不增加推理参数或改变原始前向动力学。
6. 干预及实验完整性检查全部通过。

采用2000次样本级配对Bootstrap，seed=2027。

报告准确率及交叉熵差异和95% CI。

不将样本级CI解释为跨训练种子稳定性。

若G0-S或G1-S获得良好性能但M4-S未进一步改善，则不能把收益归因于ARTG特有梯度路径。

若三个sum组均优于历史mean结果，只能初步提示尺度恢复可能有价值；历史比较仍应考虑不同执行版本和训练随机过程等限制。

若本轮所有方法均未超过简单C0，则不进入完整200轮训练。

30%结果只作为探索性辅助结果。

本轮禁止根据实际结果事后修改判定门槛。

## 十四、资源预算及停止条件

需要用户新授权。

建议：

- 1张符合实验室共享管理规定的RTX4090。
- GPU总占用上限60分钟。
- CPU累计计算上限90分钟。
- 前置检查及短程干预诊断优先。
- 仅在EXP-012全部门槛通过后训练三组各5轮。
- 不开展超参数搜索。
- 不额外增加完整训练组。

复用原可靠进程管理、GPU隔离、预算截止、心跳及原子检查点机制。

出现以下情况停止：

- 干预有效性门槛失败。
- 梯度公式检查失败。
- 非有限损失或梯度。
- 阈值非正。
- 前向动力学不一致。
- 数据校验失败。
- OOM。
- 资源预算超限。

保存全部历史失败记录。

不得未经批准增加Epoch或自动尝试其他梯度缩放方法。

## 十五、相关文献与创新性风险

LITERATURE.md追加：

1. Esser et al., Learned Step Size Quantization, ICLR 2020。

https://arxiv.org/abs/1902.08153

2. Zenke and Vogels, The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks, 2021。

https://pmc.ncbi.nlm.nih.gov/articles/PMC7619566/

3. Learnable Surrogate Gradient for Direct Training Spiking Neural Networks, IJCAI 2023。

https://ijcai.org/proceedings/2023/335

4. Stabilizing Direct Training of Spiking Neural Networks: Membrane Potential Initialization and Threshold-robust Surrogate Gradient。

https://arxiv.org/abs/2511.08708

记录共享参数梯度缩放及替代梯度尺度已有相关研究。

不宣称mean改sum本身是独立学术创新。

## 十六、科研状态更新

新增research/ROUND-012-TASK.md。

THEORY.md追加v0.1.8及R40—R46。

HYPOTHESES.md：

- 保留H-001至H-010历史结论。
- H-011更新为INSUFFICIENT_EVIDENCE，注明原筛选失败和干预不可分辨。
- H-012新增UNTESTED。
- H-013新增UNTESTED。

EXP-011保持COMPLETED，先导筛选失败。

新增EXP-012、EXP-013。

DECISION_LOG.md追加DEC-012：

不立即淘汰ARTG；先修正阈值任务梯度干预尺度，只有有效更新得到验证后才重新进行性能实验。

## 十七、结果交付

建立EXP-012及EXP-013独立目录和唯一Run ID。

至少保存：

- TASK.md
- config.yaml或config.json
- environment.txt
- code_version.json
- commands.sh
- status.json
- gradient_aggregation_tests.json
- mixed_precision_tests.json
- real_optimizer_step_check.json
- threshold_shapes.csv
- threshold_step_diagnostics.csv
- short_trajectory_comparison.csv
- intervention_gate.json
- epoch_metrics.csv
- threshold_update_summary.csv
- threshold_gradient_statistics.csv
- candidate_history.csv
- selection_lock.json
- validation_per_sample.csv
- validation_batch_sops.csv
- summary.csv
- metrics.json
- checkpoint_manifest.json
- run.log
- REPORT.md
- limitations.md
- figures/

未执行的实验项目保持NA，并说明原因。

图表至少包含：

1. mean与sum阈值梯度量级比较。
2. FP32 ULP及实际optimizer更新量比较。
3. 短程阈值更新轨迹。
4. 各层任务更新与手动更新的竞争关系。
5. 三组训练曲线。
6. 各组真实Accuracy–SOPs权衡。

后四项涉及未运行实验时按实际数据交付，不生成虚构图表。

## 十八、GitHub交付

正式运行前提交任务、固定配置和实现代码。

完成后提交全部真实结果、日志、失败记录、报告及科研状态。

保存实际运行代码SHA和结果提交SHA。

大型权重保留服务器，记录路径、大小及SHA256。

不修改main，不force push，不覆盖原Baseline及历史实验。

最终返回：

- GitHub实验分支链接。
- 执行代码SHA。
- 结果提交SHA。
- EXP-012状态。
- 实际各层N。
- mean与sum数值一致性测试。
- 各组实际非零阈值更新模块数。
- 短程阈值轨迹是否可分辨。
- H-012干预门槛是否通过。
- EXP-013是否执行。
- 正式训练各组完成Epoch。
- 各组未经校准及同预算真实性能。
- M4-S与G0-S/G1-S/C0的配对差异及CI。
- 各组最终模型状态差异。
- 实际GPU及CPU资源消耗。
- 全部异常和局限。
- 报告与原始数据路径。

本轮目标不是强行证明ARTG有效，而是确保实验真正改变阈值任务学习之后，再判断不同代理梯度是否具有独立性能价值。

获得用户资源授权后直接执行EXP-012；只有全部预注册进入条件通过才执行EXP-013。