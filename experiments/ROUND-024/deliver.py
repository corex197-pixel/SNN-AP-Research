"""Summarize saved EXP-026 evidence; CPU-only, no new inference."""
from pathlib import Path
import json,csv,time,resource,math,os
os.environ['MPLBACKEND']='Agg';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parents[2];E=P/'experiments/EXP-026';R=P/'research';start=time.monotonic();before=resource.getrusage(resource.RUSAGE_SELF)
def read(name):return json.loads((E/name).read_text())
def rows(name):
 with (E/name).open() as f:return list(csv.DictReader(f))
def writejson(name,x):(E/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def append(name,s):
 with (R/name).open('a') as f:f.write('\n\n'+s+'\n')
s=read('status.json');assert s['status'] in ('COMPLETED','PARTIAL'); val=rows('validation_summary.csv') if (E/'validation_summary.csv').exists() else [];vv={r['group']:r for r in val};lock=read('lock_commit.json') if (E/'lock_commit.json').exists() else None;gate=read('hypothesis_gate_results.json') if (E/'hypothesis_gate_results.json').exists() else None;recon=rows('reconstruction_summary.csv') if (E/'reconstruction_summary.csv').exists() else [];rr={(r['group'],r['split']):r for r in recon};boot=read('paired_bootstrap.json') if (E/'paired_bootstrap.json').exists() else None;support=read('readout_support_audit.json') if (E/'readout_support_audit.json').exists() else None
if s['status']=='COMPLETED':
 fig=E/'figures';fig.mkdir(exist_ok=True)
 def save(name):plt.tight_layout();plt.savefig(fig/name,dpi=150);plt.close()
 plt.figure(figsize=(7,4));
 for g in ['B0','50-R1','50-R2','50-R3','75-R1','75-R2','75-R3']:
  x=vv[g];plt.scatter(float(x['compression_fraction'])*100,float(x['accuracy'])*100,label=g)
 plt.xlabel('Physical parameter compression (%)');plt.ylabel('Validation accuracy (%)');plt.legend(fontsize=7,ncol=2);save('compression_accuracy.png')
 plt.figure(figsize=(8,4));labs=[];ys=[]
 for ratio in ['50','75']:
  for group in ['R0-20261128','R0-20261129','R0-20261130','R1','R2']:
   labs.append(ratio+' '+group);ys.append(float(vv[ratio+'-'+group]['accuracy'])*100)
 plt.bar(range(len(ys)),ys);plt.xticks(range(len(ys)),labs,rotation=55,ha='right',fontsize=7);plt.ylabel('Accuracy (%)');save('random_l1_fr_accuracy.png')
 plt.figure(figsize=(7,4));labs=[];ys=[]
 for ratio in ['50','75']:
  for split in ['D_grad','D_cal']:
   for group in ['R1','R3']:
    labs.append(f'{ratio} {split} {group}');ys.append(float(rr[(ratio+'-'+group,split)]['mse']))
 plt.bar(range(len(ys)),ys);plt.xticks(range(len(ys)),labs,rotation=55,ha='right',fontsize=7);plt.ylabel('Teacher logits MSE');save('reconstruction_mse.png')
 plt.figure(figsize=(6,4));labs=['50-R1','50-R3','75-R1','75-R3'];plt.bar(labs,[float(vv[g]['accuracy'])*100 for g in labs]);plt.ylabel('Accuracy (%)');save('equal_budget_accuracy.png')
 plt.figure(figsize=(7,3.5));keys=['50-R1 minus B0','50-R3 minus 50-R1','75-R1 minus B0','75-R3 minus 75-R1']
 for i,k in enumerate(keys):
  b=boot['comparisons'][k];lo,hi=b['accuracy_ci95_pp'];x=b['accuracy_delta_pp'];plt.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',capsize=3)
 plt.axvline(0,color='black',linewidth=.8);plt.yticks(range(len(keys)),keys);plt.xlabel('Paired accuracy difference (percentage points)');save('paired_accuracy_ci.png')
 plt.figure(figsize=(7,4));
 for g in ['B0','50-R1','50-R3','75-R1','75-R3']:
  x=vv[g];plt.scatter(float(x['parameter_count'])/1e6,float(x['SOP_sample_weighted'])/1e6,label=g)
 plt.xlabel('Physical parameters (millions)');plt.ylabel('Theoretical SOP/sample (millions)');plt.legend();save('parameters_sops.png')
# Resource report includes measured main process tree CPU and a separate explicitly unmeasured early audit reserve.
after=resource.getrusage(resource.RUSAGE_SELF);delivery_cpu=(after.ru_utime-before.ru_utime)+(after.ru_stime-before.ru_stime);early=60.;run1=read('runs/EXP-026-RUN-001/status.json');run2_cpu=s.get('process_tree_cpu_seconds');cpu_known=(run1.get('process_tree_cpu_seconds') or 0)+(run2_cpu or 0)+delivery_cpu;budgeted=(s.get('cpu_accounted_seconds') or 0)+delivery_cpu
with (E/'resource_usage.csv').open('w',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['scope','metric','value','unit','basis','limitation']);w.writerow(['RUN-001','gpu_process_wall',run1['wall_seconds'],'seconds','conservative process clock','not GPU kernel active time']);w.writerow(['RUN-002','gpu_process_wall',s['wall_seconds']-run1['wall_seconds'],'seconds','conservative process clock','includes lock push and CPU work']);w.writerow(['both','gpu_budget_wall',s['wall_seconds'],'seconds','sum of process clocks','initial imports before clock not included; CUDA not intentionally initialized there']);w.writerow(['RUN-001+002','process_tree_cpu',cpu_known-delivery_cpu,'seconds','getrusage plus live descendants','timing begins after imports; sampling process trees can miss races']);w.writerow(['delivery','cpu',delivery_cpu,'seconds','getrusage self delta','does not include other command processes']);w.writerow(['prelaunch','cpu_reserve',early,'seconds','conservative allowance','estimate not measurement']);w.writerow(['total','cpu_budget_accounted',budgeted,'seconds','run tree CPU plus 60s reserve plus delivery CPU','early commands, imports and remote tool CPU not fully measured'])

# Preserve the raw automatic gate separately; the selected-model pre-evaluation
# coverage was not verified and cannot be retroactively asserted.
if gate:
 for q in gate['ratios'].values():
  q['H027_checks']['admission_and_isolation']=False
  q['H027_all']=False
  q['admission_limitation']='selected 50% and 75% R1/R3 hidden-spike equality not verified before evaluation'
 writejson('hypothesis_gate_results.json',gate)

lines=['# EXP-026 / ROUND-024 结果报告',f"原进程状态：{s['status']}；正式准入覆盖不完整，正式实验执行状态 INVALID，数值仅供探索；实际运行代码 `{s['code_sha']}`；选择锁 `{lock['sha'] if lock else 'NA'}`。父检查点及固定划分、严格 FP32、0 SGD、单 GPU，详细配置见 TASK.md/config.json。",'','## 固定验证集结果','Accuracy 单位为百分比；SOP 为理论样本加权计数，FR 为全网络。','', '| 组 | Accuracy % | CE | FR | SOP/sample | 物理参数 |','|---|---:|---:|---:|---:|---:|']
for v in val:lines.append(f"| {v['group']} | {100*float(v['accuracy']):.5f} | {float(v['CE']):.6f} | {float(v['FR_global']):.6f} | {float(v['SOP_sample_weighted']):.2f} | {int(v['parameter_count'])} |")
lines+=['','## 重构误差','| 组 | 样本 | logits MSE | RMSE | 教师预测一致率 | CE |','|---|---|---:|---:|---:|---:|']
for x in recon:lines.append(f"| {x['group']} | {x['split']} | {float(x['mse']):.8g} | {float(x['rmse']):.8g} | {float(x['teacher_prediction_agreement']):.6f} | {float(x['CE']):.6f} |")
if gate:
 lines+=['','## 预注册门槛','```json',json.dumps(gate,ensure_ascii=False,indent=2),'```','假设科学状态保持 UNTESTED，等待理论研究员审核。75% 为辅助结果；拟合误差下降不等同验证性能恢复。']
if boot:
 lines+=['','## 配对 Bootstrap','2000 次、seed=20261128、样本配对百分位区间；逐样本 SOP 区间 NA。随机集合不是独立训练种子。']
 for k in ['50-R1 minus B0','50-R3 minus 50-R1','75-R1 minus B0','75-R3 minus 75-R1']:
  q=boot['comparisons'][k];lines.append(f"- {k}: accuracy {q['accuracy_delta_pp']:.5f} pp, 95% CI {q['accuracy_ci95_pp']}; CE {q['CE_delta']:.7f}, CI {q['CE_ci95']}.")
lines+=['','## 检查、资源和局限',f"RUN-001 在准入测试计数断言停止，7 项新增检查实际均通过；错误和原日志完整保留。RUN-002 修正计数后重新执行准入，未调整科学公式和门槛。选择锁远程验证后才使用 D_cal 和验证集。GPU 进程墙钟累计 {s['wall_seconds']:.3f} 秒；已测进程树 CPU 与交付脚本约 {cpu_known:.3f} 秒，加早期未测 CPU 保守预留后预算计数 {budgeted:.3f} 秒，见 resource_usage.csv；预留不是实测值。",'严格 FP32 B0 的准确率、CE 和 SOP 与 EXP-024 完全一致；完整参数及哈希参见相关 audit/manifest。没有使用官方测试集，没有训练。预检查的 R120 脉冲一致性为固定人工删除集合的真实网络检查；每个实际选择集合未独立再做脉冲逐元素比对，见 limitations.md。','H-026/H-027 仅以观察门槛报告；不能宣称闭式重构是首创或形成独立统计确认。']
(E/'REPORT.md').write_text('\n'.join(lines)+'\n')
(E/'limitations.md').write_text('''# 局限与审计\n\n- 同一父模型和单一开发验证集，随机集合并非独立训练种子。75% 是辅助。\n- 正式测试集未读取；D_grad/D_cal 均来自父模型训练集，D_cal 仅对本次拟合独立。\n- RUN-001 的预检汇总检查数量错误，真实检查均已执行，已停止并保存；RUN-002 完全重做且累计资源。\n- 准入前 R120 在真实网络、固定人工删除 1024 个神经元及人工非零补偿下检查全部非目标状态、隐藏脉冲及 logits；并未逐一对最终选择的 50% 和 75% R1/R3 模型执行独立的脉冲逐元素比较。这一验证范围缺口使对应严格准入条件不能被视作完整通过；最终性能为已执行的探索性结果，不可视作全部门槛已满足。\n- CPU 进程树计数开始于 Python 导入之后；其他审查命令未全程计数，早期 60 秒为预算预留而非测量。GPU 墙钟是受控进程时长的保守代理，不是硬件能耗或核执行时间。\n- 如无逐样本 SOP，SOP 配对区间标记 NA。\n''')
# Do not leave H027's synthetic admission gate as a claimed full pass for selected models.
if gate:
 for q in gate['ratios'].values():q['H027_checks']['admission_and_isolation']=False;q['H027_all']=False;q['admission_limitation']='preselection synthetic retained-spike check only; final selected R1/R3 pairs not independently elementwise verified'
 writejson('hypothesis_gate_results.json',gate)
(E/'run.log').write_text('RUN-001: runs/EXP-026-RUN-001/run.log\nRUN-002: runs/EXP-026-RUN-002/run.log\n')
append('RESEARCH_STATE.md',f"ROUND-024 EXP-026 execution {s['status']}: baseline reconstructed; preselection check coverage limitation retained; H-026/H-027 remain UNTESTED pending theoretical review. No official test or SGD. See ../experiments/EXP-026/REPORT.md.")
with (R/'EXPERIMENT_REGISTRY.csv').open('a',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['EXP-026','EXP-026-RUN-001','ROUND-024','admission','H-026;H-027','v0.3.0','research/round-024-structural-reconstruction',run1['code_sha'],'','','PARTIAL','UNTESTED','experiments/EXP-026/config.json','','experiments/EXP-026/REPORT.md','','',run1['wall_seconds'],'seven passed checks miscounted as six; no selection']);w.writerow(['EXP-026','EXP-026-RUN-002','ROUND-024','fixed support reconstruction','H-026;H-027','v0.3.0','research/round-024-structural-reconstruction',s['code_sha'],'','','INVALID','INVALID_EVIDENCE','experiments/EXP-026/config.json','experiments/EXP-026/validation_summary.csv','experiments/EXP-026/REPORT.md','','',s['wall_seconds'],'per-selected-group spike equality not independently tested; see limitations'])
print('delivery_cpu_seconds',delivery_cpu,'budget_accounted',budgeted)
