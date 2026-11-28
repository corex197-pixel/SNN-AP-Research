"""CPU-only summary of actual locked EXP-028 outputs."""
from pathlib import Path
import csv,json,resource,os,datetime
os.environ.update(MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[2];E=P/'experiments/EXP-028';R=P/'research';before=resource.getrusage(resource.RUSAGE_SELF)
def j(n):return json.loads((E/n).read_text())
def rows(n):
 with (E/n).open() as f:return list(csv.DictReader(f))
def app(n,s):
 with (R/n).open('a') as f:f.write('\n\n'+s+'\n')
s=j('status.json');run=j('runs/EXP-028-RUN-004/status.json');assert s['status']=='COMPLETED' and s['validity']=='VALID' and run['completed_epochs']=={'F0':5,'F1':5};tlock=j('training_admission_lock_commit.json');elock=j('evaluation_lock_commit.json');gate=j('hypothesis_gate_results.json');boots=j('paired_bootstrap.json');v=rows('validation_summary.csv');vr={x['group']:x for x in v};ep=rows('epoch_metrics.csv');tr=rows('topology_transition.csv');frozen=j('frozen_parameter_audit.json');manifest=j('checkpoint_manifest.json');hist=j('historical_reproduction_audit.json');neg=j('admission_negative_tests.json');assert all(x['passed'] for x in frozen.values()) and neg['all_passed'];assert all(all(y==0 for y in x.values()) for x in hist.values());assert len(ep)==10
fig=E/'figures';fig.mkdir(exist_ok=True)
def save(n):plt.tight_layout();plt.savefig(fig/n,dpi=145);plt.close()
for field,title,name in [('train_CE','Training CE','training_CE.png'),('train_accuracy','Training accuracy','training_accuracy.png')]:
 plt.figure(figsize=(6,3.5))
 for g in ['F0','F1']:
  a=[x for x in ep if x['group']==g];plt.plot([int(x['epoch']) for x in a],[float(x[field]) for x in a],marker='o',label=g)
 plt.xlabel('Epoch');plt.ylabel(title);plt.legend();save(name)
plt.figure(figsize=(6,3.5));names=['B0','P0','F0','F1'];plt.bar(names,[100*float(vr[g]['accuracy']) for g in names]);plt.ylabel('Validation accuracy (%)');save('endpoint_accuracy.png')
plt.figure(figsize=(6,3.5));
for g in names:plt.scatter(int(vr[g]['parameter_count'])/1e6,float(vr[g]['SOP_sample_weighted'])/1e6,label=g)
plt.xlabel('Physical parameters (millions)');plt.ylabel('Theoretical SOP/sample (millions)');plt.legend();save('parameters_sops.png')
plt.figure(figsize=(5,3.5));plt.bar(['F0','F1'],[gate['recovery_ratio'][g] for g in ['F0','F1']]);plt.ylabel('Fraction of accuracy loss recovered');save('recovery_fraction.png')
plt.figure(figsize=(7,3.5));
for i,k in enumerate(['F0-P0','F1-P0','F1-F0','F0-B0','F1-B0']):
 x=boots['comparisons'][k];mid=x['accuracy_delta_pp'];lo,hi=x['accuracy_ci95_pp'];plt.errorbar(mid,i,xerr=[[mid-lo],[hi-mid]],fmt='o',capsize=3)
plt.axvline(0,color='black',lw=.8);plt.yticks(range(5),['F0-P0','F1-P0','F1-F0','F0-B0','F1-B0']);plt.xlabel('Paired accuracy difference (pp)');save('paired_accuracy_ci.png')
plt.figure(figsize=(8,4));z=[x for x in tr if x['group']=='F1' and x['epoch']=='5'];plt.bar(range(len(z)),[int(x['activated'])-int(x['deactivated']) for x in z]);plt.xticks(range(len(z)),[x['layer'] for x in z],rotation=45,ha='right',fontsize=7);plt.ylabel('Net effective connections vs P0');save('layer_connections.png')
summary={}
for g in ['F0','F1']:
 z=[x for x in tr if x['group']==g and x['epoch']=='5'];q=[x for x in ep if x['group']==g];summary[g]={'activated':sum(int(x['activated']) for x in z),'deactivated':sum(int(x['deactivated']) for x in z),'train_seconds':sum(float(x['train_seconds']) for x in q),'successful_steps':sum(int(x['successful_steps']) for x in q),'amp_skips':sum(int(x['amp_skips']) for x in q)}
(E/'topology_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# EXP-028 结构化剪枝后任务微调强对照','',f"训练 RUN-004：F0/F1 各 5 epoch；修复后终点评价 EVAL-001：{s['status']}，VALID。训练代码 `{s['training_code_sha']}`，评价代码 `{s['code_sha']}`。训练锁 `{tlock['sha']}`，修复后的终点锁 `{elock['sha']}`。",'','## 结果','| 组 | Accuracy % | CE | FR_global | SOP/sample | 参数 |','|---|---:|---:|---:|---:|---:|']
for x in v:lines.append(f"| {x['group']} | {100*float(x['accuracy']):.5f} | {float(x['CE']):.8f} | {float(x['FR_global']):.8f} | {float(x['SOP_sample_weighted']):.2f} | {x['parameter_count']} |")
lines+=['',f"R126：F0={gate['recovery_ratio']['F0']:.6f}，F1={gate['recovery_ratio']['F1']:.6f}；分母={gate['recovery_denominator']:.8f}。F1 SOP≤101% P0：{gate['F1_SOP_le_101pct_P0']}。SOP 为理论计数，不是实际能耗。",'','## 固定门槛与配对统计','```json',json.dumps(gate,indent=2,ensure_ascii=False),'```','H-028/H-029 最终科研状态由理论研究员确定。']
for k,x in boots['comparisons'].items():lines.append(f"- {k}: accuracy {x['accuracy_delta_pp']:.5f} pp，95% CI {x['accuracy_ci95_pp']}；CE {x['CE_delta']:.8f}，95% CI {x['CE_ci95']}；参考错候选对 {x['reference_wrong_candidate_right']}，参考对候选错 {x['reference_right_candidate_wrong']}；预测差异 {x['prediction_disagree']}。")
lines+=['','## 训练与结构保护']
for g,x in summary.items():lines.append(f"- {g}：5 轮、实际成功更新 {x['successful_steps']}、AMP 跳步 {x['amp_skips']}、训练记录耗时 {x['train_seconds']:.2f} s；相对 P0 新增有效连接 {x['activated']}、失活 {x['deactivated']}。")
lines+=['F0 每轮对所有 66 个非目标参数及持久化 buffer 逐项核查，差异均为零。两组阈值保持固定，手动惩罚调用为零；实际样本顺序与丢弃索引按轮一致。B0/P0 的重新计算指标与 EXP-027 一致。','', '## 异常与局限','RUN-001/002/003 的 F0 AMP CE 预检以显式 FP32/FP64 数学参考设置的 1e-6 门槛失败，未启动正式训练；匹配原始 AMP 公式的重复结果一致，高精度差异约 1.375e-5，见两个只读诊断。原结果、异常和资源均保留。','RUN-004 的终点评价前拦截失败，先完成了 B0 评价并开始 P0；未评价 F0/F1 后被人工停止。原终点锁及这部分评价无效，已归档。EVAL-001 使用既有最终检查点，在真实恢复模型前向与加载纯推理权重后的前向完全一致、其他终点条件核查通过且新锁远程确认后，重新评价四组。没有重训或更改候选。','验证集已多轮开发使用，D_grad 属于父模型训练集；一个父训练种子及一个微调种子不能推断跨种子稳定性。样本级 Bootstrap 并非训练重复。可选 R123 margin 诊断未采集 logits，标记 NA；未因此增加 GPU 计算。',f"GPU 受控进程墙钟（含早期预检、诊断及评价，保守累计）{s['wall_seconds']:.2f} s；CPU 已测进程树与启动前预留合计约 {s['cpu_budget_accounted_seconds']:.2f} s（尚加交付脚本时间），120 s 启动前预留为估计而非实测，完整局限见 resource_usage.csv。GPU 墙钟不是 GPU 内核时间或能耗。"]
(E/'REPORT.md').write_text('\n'.join(lines)+'\n');(E/'limitations.md').write_text('''# 限制与偏差\n\n- RUN-001/002/003 的自设 AMP 与高精度 CE 对照未通过；相同原始 AMP 公式重复一致，FP64 差异约 1.375e-5 单独记录，不改变训练算法。\n- RUN-004 终点评价前拦截器未生效，完整 B0 及部分 P0 在未充分覆盖的旧锁后被读取；该旧评价及旧终点锁无效。F0/F1 验证未在停止前启动。EVAL-001 在补充真实 recovery-to-inference 对象检查、远程新锁和入口校验后独立重新评价。\n- 固定验证集多次参与开发，只有一个父模型及微调种子；Bootstrap 是配对样本重采样，不是独立训练重复。\n- R123 辅助诊断 NA：没有保存评价 logits，未追加 GPU 运算。SOP 配对区间 NA。\n- CPU 进程树监测从 Python 导入后开始；早期 shell、工具和外部进程没有全覆盖。120s 为预算预留而不是实测。GPU 墙钟含 CPU 和网络等待，非精确内核时间或能耗。\n''');(E/'run.log').write_text('RUN-001/002/003: runs/EXP-028-RUN-00N/run.log\nRUN-004: runs/EXP-028-RUN-004/run.log (training complete; first endpoint evaluation invalid)\nEVAL-001: runs/EXP-028-EVAL-001/run.log (repaired endpoint admission and independent evaluation)\n')
cv=j('code_version.json');cv.update(training_code_sha=s['training_code_sha'],evaluation_code_sha=s['code_sha'],training_admission_lock=tlock['sha'],evaluation_admission_lock=elock['sha']);(E/'code_version.json').write_text(json.dumps(cv,indent=2)+'\n')
app('RESEARCH_STATE.md','ROUND-026 EXP-028: F0/F1 each completed 5 epochs; training admission passed; original RUN-004 endpoint evaluation INVALID due missing pre/post-load check and premature B0/partial P0 evaluation. Existing checkpoints were not retrained; EVAL-001 completed with repaired candidate-specific evaluation lock and VALID evaluation. H-028/H-029 preregistered gates both false; scientific statuses remain UNTESTED pending theory review. All deviations retained in ../experiments/EXP-028/REPORT.md.')
with (R/'EXPERIMENT_REGISTRY.csv').open('a',newline='') as f:
 w=csv.writer(f,lineterminator='\n')
 for runid in ['001','002','003']:
  x=j(f'runs/EXP-028-RUN-{runid}/status.json');w.writerow(['EXP-028',f'EXP-028-RUN-{runid}','ROUND-026','AMP preflight','H-028;H-029','v0.3.2','research/round-026-compact-finetuning',x['code_sha'],'','','PARTIAL','UNTESTED','experiments/EXP-028/config.json','','experiments/EXP-028/REPORT.md','','',x['wall_seconds'],'F0 CE precision check failed; no formal training'])
 w.writerow(['EXP-028','EXP-028-RUN-004','ROUND-026','fixed training and initial invalid evaluation','H-028;H-029','v0.3.2','research/round-026-compact-finetuning',run['code_sha'],'','','PARTIAL','INVALID_EVIDENCE','experiments/EXP-028/config.json','experiments/EXP-028/epoch_metrics.csv','experiments/EXP-028/REPORT.md','','',run['wall_seconds'],'training 5+5 complete; old endpoint lock incomplete and early B0/partial P0 evaluation invalid'])
 w.writerow(['EXP-028','EXP-028-EVAL-001','ROUND-026','repaired endpoint admission and evaluation','H-028;H-029','v0.3.2','research/round-026-compact-finetuning',s['code_sha'],'','','COMPLETED','UNTESTED','experiments/EXP-028/config.json','experiments/EXP-028/validation_summary.csv','experiments/EXP-028/REPORT.md',datetime.datetime.fromtimestamp(s['start_unix'],datetime.timezone.utc).isoformat(),datetime.datetime.fromtimestamp(s['end_unix'],datetime.timezone.utc).isoformat(),s['wall_seconds'],'new endpoint lock remote; first incomplete evaluation preserved'])
after=resource.getrusage(resource.RUSAGE_SELF);cpu=after.ru_utime-before.ru_utime+after.ru_stime-before.ru_stime
with (E/'resource_usage.csv').open('w',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['scope','metric','value','unit','basis','limitation']);w.writerow(['all cumulative','gpu_process_wall',s['wall_seconds'],'seconds','prior runs and two diagnostics plus RUN-004 and EVAL-001','includes CPU/network; excludes initial Python imports in process clocks; not GPU kernel time']);w.writerow(['RUN-004','process_tree_cpu',run['process_tree_cpu_seconds'],'seconds','getrusage and live descendants','sampling can race and starts after imports']);w.writerow(['EVAL-001','process_tree_cpu',s['process_tree_cpu_seconds'],'seconds','getrusage and live descendants','starts after imports']);w.writerow(['earlier preflights and diagnostics','cpu_including_prelaunch_reserve',run['cpu_prior_accounted_seconds'],'seconds','measured prior processes plus 120s estimate','prelaunch reserve not measurement']);w.writerow(['delivery','self_cpu',cpu,'seconds','getrusage delta','other shell and service CPU excluded']);w.writerow(['total','cpu_budget_accounted',s['cpu_budget_accounted_seconds']+cpu,'seconds','cumulative run counts plus reserve and delivery','early tools CPU not fully measured'])
print('delivery_cpu',cpu,'total_budget_accounted',s['cpu_budget_accounted_seconds']+cpu)
