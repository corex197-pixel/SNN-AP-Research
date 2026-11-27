"""CPU-only delivery for completed or safely stopped EXP-018/019."""
import os,sys,time,csv,json,hashlib,statistics,resource,subprocess,traceback
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];A=R/'experiments/EXP-018';B=R/'experiments/EXP-019';start=time.monotonic();cpu0=resource.getrusage(resource.RUSAGE_SELF)
def read(p):
 if not p.exists():return []
 with p.open(newline='') as f:return list(csv.DictReader(f))
def dump(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
status=json.loads((A/'status.json').read_text());rows=read(A/'epoch_metrics.csv');cost=read(B/'total_cost_decomposition.csv');layers=read(B/'layer_cost_decomposition.csv');gates=json.loads((A/'stage_gate_results.json').read_text()) if (A/'stage_gate_results.json').exists() else {};by={g:[x for x in rows if x['group']==g] for g in ['U0','U2']};cost_epochs=[int(x['epoch']) for x in cost];cost_state='COMPLETED' if cost_epochs==[20,60,100,200,300] else 'PARTIAL';primary=next((x for x in cost if int(x['epoch'])==60),None);dump(B/'status.json',{'execution_status':cost_state,'run_id':'EXP-019-COST-001','recorded_epochs':cost_epochs,'primary_epoch':60,'H018_scientific_status':'UNTESTED_pending_theory_review','code_sha':status.get('code_sha'),'limitations':'paired train-origin validation; counterfactual counts do not establish training causality'})
manifest=[]
for g in ['U0','U2']:
 d=A/'runs/EXP-018-CONTINUE-001'/g
 for n in ['latest.pth','best_validation.pth','epoch-060.pth','epoch-100.pth','epoch-200.pth','epoch-300.pth']:
  p=d/n
  if p.exists():manifest.append({'group':g,'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'note':'epoch and code in checkpoint; full weights stay on server'})
dump(A/'checkpoint_manifest.json',manifest)
# Recheck paired sample-order hashes where both groups completed same epoch.
audit=read(A/'training_sample_audit.csv');pairs=[]
for e in sorted({int(x['epoch']) for x in audit}):
 x=[v for v in audit if int(v['epoch'])==e]
 if len(x)==2:pairs.append({'epoch':e,'same_order':len({v['sample_order_sha256'] for v in x})==1,'same_dropped':len({v['dropped_ids'] for v in x})==1})
dump(A/'paired_sampling_audit.json',{'pairs':pairs,'all_passed':all(v['same_order'] and v['same_dropped'] for v in pairs),'unpaired_epochs':len({int(x['epoch']) for x in audit})-len(pairs)})
(A/'figures').mkdir(exist_ok=True);(B/'figures').mkdir(exist_ok=True)
def save(path):plt.tight_layout();plt.savefig(path,dpi=120);plt.close()
for name,key,label in [('accuracy.png','validation_accuracy','Validation accuracy'),('loss.png','validation_loss','Validation loss'),('sops.png','SOP_sample_weighted','SOP / sample')]:
 plt.figure(figsize=(8,4))
 for g,a in by.items():
  if a:plt.plot([int(x['epoch']) for x in a],[float(x[key]) for x in a],label=g)
 plt.xlabel('Epoch');plt.ylabel(label)
 if rows:plt.legend()
 save(A/'figures'/name)
plt.figure(figsize=(7,4))
for g,a in by.items():
 if a:plt.plot([float(x['SOP_sample_weighted']) for x in a],[float(x['validation_accuracy']) for x in a],label=g)
if rows:plt.legend()
plt.xlabel('SOP / sample');plt.ylabel('Validation accuracy');save(A/'figures/accuracy_vs_sops.png')
for source,key,name in [('layer_sparsity.csv','density','layer_density.png'),('raw_weight_statistics.csv','threshold_to_median','threshold_weight_ratio.png')]:
 r=read(A/source);plt.figure(figsize=(10,4))
 for g in by:
  for n in sorted({x['layer'] for x in r if x['group']==g}):
   x=[a for a in r if a['group']==g and a['layer']==n and a[key]!='NA_zero_median']
   if x:plt.plot([int(a['epoch']) for a in x],[float(a[key]) for a in x],label=g+' '+n)
 if r:plt.legend(fontsize=5,ncol=3)
 plt.xlabel('Epoch');plt.ylabel(key);save(A/'figures'/name)
r=read(A/'effective_fanout_statistics.csv');plt.figure(figsize=(8,4))
for g in by:
 for n in sorted({x['neuron'] for x in r if x['group']==g}):
  v=[x for x in r if x['group']==g and x['neuron']==n];plt.plot([int(x['epoch']) for x in v],[float(x['mean']) for x in v],marker='o',label=g+' '+n)
if r:plt.legend(fontsize=6)
plt.xlabel('Epoch');plt.ylabel('Mean effective fanout');save(A/'figures/fanout.png')
for name,keys,y in [('four_costs.png',['C00','C20','C02','C22'],'SOP / sample'),('cost_components.png',['delta_W','delta_S','delta'],'SOP change / sample'),('relative_components.png',['relative_W','relative_S','relative_total'],'Relative savings')]:
 plt.figure(figsize=(8,4))
 for k in keys:
  if cost:plt.plot([int(x['epoch']) for x in cost],[float(x[k]) for x in cost],marker='o',label=k)
 if cost:plt.legend()
 plt.axhline(0,color='black',lw=.5);plt.xlabel('Epoch');plt.ylabel(y);save(B/'figures'/name)
lines=['# EXP-018：Controlled UniSparse 分阶段续训','',f"状态：{status['status']}；执行代码 `{status['code_sha']}`；完成轮次 {status['completed']}。",'', '| 阶段 | U0 准确率 | U2 准确率 | U0 SOP | U2 SOP | U2 FR | U2 受惩罚稀疏率 | 门槛 |','|---|---:|---:|---:|---:|---:|---:|---|']
for stage,x in gates.items():
 a=x['U0'];b=x['U2'];lines.append('| {} | {:.6f} | {:.6f} | {:.2f} | {:.2f} | {:.6f} | {:.6f} | {} |'.format(stage,a['validation_accuracy'],b['validation_accuracy'],a['SOP_sample_weighted'],b['SOP_sample_weighted'],b['FR_sample_weighted'],b['penalized_weight_sparsity'],x['conditions']['passed']))
lines+=['',f"停止原因/异常：{status.get('stop_reason',status.get('error','无'))}。资源保守墙钟 {status.get('wall_seconds','NA')} 秒；累计 CPU 记录 {status.get('cpu_seconds','NA')} 秒。墙钟含前置和推送等待；CPU 计数可能漏极短生命周期子进程及进程启动。",'', '父检查点只读；结果为单种子、训练来源验证集上的 Controlled Baseline，不是论文精确复现。最佳验证模型为次要指标，固定第300轮才是完整终点。各轮采样、验证分片、逐层统计、检查点哈希及异常见本目录。']
(A/'REPORT.md').write_text('\n'.join(lines)+'\n');(A/'limitations.md').write_text('单训练种子及已用于工作点选择的验证集；官方 lambda 未确认；CPU 子进程计数可能不完整；无官方测试集；理论 SOP 不是硬件能耗。\n')
lines=['# EXP-019：反事实连接—活动成本分解','',f'执行状态：{cost_state}；有效轮次：{cost_epochs}；主要早期判定轮次固定为60。','', '| 轮次 | C00 | C20 | C02 | C22 | ΔC_W | ΔC_S | I_WS | ΔC_S/C00 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for x in cost:lines.append('| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.4%} |'.format(x['epoch'],*[float(x[k]) for k in ['C00','C20','C02','C22','delta_W','delta_S','interaction','activity_cost_fraction']]))
lines+=['',f"第60轮主要方向及5%探索幅度：{('方向='+str(primary['direction_gate'])+'，幅度='+str(primary['exploratory_magnitude_gate'])) if primary else 'NA，配对第60轮尚未完成'}。H-018 仍待理论审查；H-017 UNTESTED。",'', 'C20/C02 是固定真实输入后更换计数掩码的反事实成本，非交叉模型的真实前向。该对称分解不识别训练因果效应；所有结果基于已用于工作点选择的验证集。按样本数加权，理论 SOP 不等同实测能耗或推理速度。逐层和监控器核对见 CSV 与 cost_count_audit JSON。']
(B/'REPORT.md').write_text('\n'.join(lines)+'\n');(B/'limitations.md').write_text(lines[-1]+'\n')
with open(R/'research/RESEARCH_STATE.md','a') as f:f.write(f"\nROUND-017 delivery: EXP-018 {status['status']} at {status['completed']}; EXP-019 {cost_state} at epochs {cost_epochs}. H-016 PARTIALLY_SUPPORTED; H-017/H-018 UNTESTED pending theory review. See EXP-018/019 reports.\n")
with open(R/'research/UNISPARSE_BASELINE.md','a') as f:f.write(f"\nROUND-017 controlled continuation: status {status['status']}, completed {status['completed']}; cost checkpoints {cost_epochs}. User-specified stage gates and exact parent states preserved; see EXP-018/019. Not a paper-table reproduction.\n")
p=R/'research/EXPERIMENT_REGISTRY.csv'
with p.open(newline='') as f:r=csv.DictReader(f);fields=r.fieldnames;reg=list(r)
for x in reg:
 if x.get('experiment_id') in ['EXP-018','EXP-019']:
  x['execution_status']=status['status'] if x['experiment_id']=='EXP-018' else cost_state;x['scientific_status']='UNTESTED';x['code_sha']=status['code_sha'];x['metrics_path']='experiments/EXP-018/epoch_metrics.csv' if x['experiment_id']=='EXP-018' else 'experiments/EXP-019/total_cost_decomposition.csv';x['notes']='stage-gated continuation' if x['experiment_id']=='EXP-018' else 'counterfactual counting; primary epoch60'
with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(reg)
r=resource.getrusage(resource.RUSAGE_SELF);dump(A/'delivery_resources.json',{'wall_seconds':time.monotonic()-start,'cpu_seconds':r.ru_utime+r.ru_stime-cpu0.ru_utime-cpu0.ru_stime,'gpu_used':False});print('finalized',status['status'],cost_state)
