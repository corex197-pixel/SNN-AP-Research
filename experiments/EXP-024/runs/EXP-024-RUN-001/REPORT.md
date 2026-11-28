# EXP-024：精确脉冲贡献与真实结构化剪枝

执行状态：PARTIAL；实际代码 `70d8ba891ae82635bc459f5289884e51e337a772`；锁定提交 `NA`；异常/停止原因 `AssertionError('admission failed readout_zero_set')`。U2第300轮为唯一父模型；历史EXP-023准入偏差及负面结果保持记录。

## 机器可核验准入与结构等价性

准入状态 `RUNNING`；R103真实FP32对FP64代数重构归一化误差 `NA`；固定单个、16个及512个神经元的紧凑模型相对门控模型最大误差 `NA`。实际样本ID、分批[2,1]、断言、输出与退出状态见 admission_gate_audit.json；各锁定集合在D_cal/验证之前的实际等价性见 selected_structure_audit.json。未通过或未执行的测试不可视为通过。

## 验证结果

| 组 | Accuracy | CE | FR | SOP/sample | 实际参数量 | 结构压缩率 | 纯推理文件字节 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NA | NA | NA | NA | NA | NA | NA | NA |

各层SOP、FR、有效非零权重、稀疏率及时间见 layer_sops.csv、layer_FR.csv、validation_summary.csv；模型路径与SHA256见checkpoint_manifest.json。SOP为理论计数，不代表实测能耗或普适GPU加速。

## 联合损失与配对比较

| 组 | 联合ΔCE | 单项ΔCE之和 | 联合减单项和 |
|---|---:|---:|---:|
| NA | NA | NA | NA |

| 比较 | Accuracy差（百分点） | 95% CI | CE差 | 95% CI |
|---|---:|---:|---:|---:|
| NA | NA | NA | NA | NA |

配对Bootstrap采用PCG64、seed=20261128、2000次百分位区间；逐样本SOP未采集，其区间为NA。M5的分块内边际分数都基于同一块开始状态；联合CE重新计算，非单项和。

## 预注册门槛与限制

`{}`；整体推进 `NA`。H-024/H-025科学状态等待理论研究员审查；不可从公式正确推断性能有效。D_grad/D_cal曾参与父模型训练，验证集多轮用于开发，单种子结果非独立确认。贪心、线性读出及直接损失计分不作学术首创声明；该分解同样适用于满足条件的ANN。

单卡 `GPU-7b1fc21b-f1b8-5b4e-5188-a1a4a117ffae`；保守预算墙钟 `6.664348840713501` 秒，CPU计数 `7.071237` 秒；导入、CPU汇总及短生命周期子进程可能漏计。未执行阶段按NA处理，严禁用合成值填补正式结果。
