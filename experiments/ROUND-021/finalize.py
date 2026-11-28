"""CPU-only EXP-023 delivery; report partial runs without inventing results."""
import os,json,csv,gzip,time,datetime,hashlib
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];E=ROOT/'experiments/EXP-023';F=E/'figures';F.mkdir(exist_ok=True);begin=time.process_time()
def read(n):
 p=E/n
 return list(csv.DictReader(p.open())) if p.exists() else []
def js(n,default=None):
 p=E/n;return json.loads(p.read_text()) if p.exists() else default
def dump(n,v):(E/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def write(n,rows):
 if rows:
  with (E/n).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
def save(n):plt.tight_layout();plt.savefig(F/n,dpi=150);plt.close()
s=js('status.json',{'status':'PARTIAL','completed':{},'error':'status missing'});gate=js('hypothesis_gate_results.json',{});ex=read('exposure_statistics.csv');co=read('weight_penalty_coefficients.csv');metrics=read('epoch_metrics.csv');layers=read('layer_sparsity.csv');top=read('topology_transition.csv');manifest=read('checkpoint_manifest.csv');dump('checkpoint_manifest.json',manifest)
if ex:
 plt.figure(figsize=(9,4));r=[x for x in ex if x['epoch']=='100' and x['dataset']=='D_grad'];plt.bar([x['layer'] for x in r],[float(x['p']) for x in r]);plt.xticks(rotation=30,ha='right');plt.ylabel('exposure p');save('exposure_by_layer.png')
if co:
 plt.figure(figsize=(9,4));names=list(dict.fromkeys(x['layer'] for x in co));idx=np.arange(len(names))
 for i,g in enumerate(['W0','W1','W2']):
  vals={x['layer']:float(x['Khat']) for x in co if x['group']==g};plt.bar(idx+(i-1)*.25,[vals[n] for n in names],width=.25,label=g)
 plt.xticks(idx,names,rotation=30,ha='right');plt.ylabel('K');plt.legend();save('coefficients.png')
if metrics:
 for field,name,ylabel in [('validation_accuracy','accuracy.png','validation accuracy'),('validation_loss','loss.png','validation CE'),('SOP_sample_weighted','sops.png','SOP/sample')]:
  plt.figure(figsize=(6,4))
  for g in ['W0','W1','W2']:
   r=[x for x in metrics if x['group']==g];
   if r:plt.plot([int(x['epoch']) for x in r],[float(x[field]) for x in r],marker='o',label=g)
  plt.xlabel('epoch');plt.ylabel(ylabel);plt.legend();save(name)
if layers:
 plt.figure(figsize=(9,4))
 for g in ['W0','W1','W2']:
  r=[x for x in layers if x['group']==g and x['epoch']=='110'];
  if r:plt.plot([x['layer'] for x in r],[1-float(x['density']) for x in r],marker='o',label=g)
 plt.xticks(rotation=30,ha='right');plt.ylabel('weight sparsity');plt.legend();save('layer_sparsity.png')
if top:
 fig,ax=plt.subplots(1,2,figsize=(10,4))
 for g in ['W0','W1','W2']:
  r=[x for x in top if x['group']==g and x['epoch']=='110'];
  if r:ax[0].plot([x['layer'] for x in r],[int(x['pruned_count']) for x in r],marker='o',label=g);ax[1].plot([x['layer'] for x in r],[int(x['regrown_count']) for x in r],marker='o',label=g)
 for a in ax:a.tick_params(axis='x',rotation=35)
 ax[0].set_title('pruned');ax[1].set_title('regrown');ax[0].legend();save('topology_changes.png')
final={g:next((x for x in metrics if x['group']==g and x['epoch']=='110'),None) for g in ['W0','W1','W2']};completed=all(final.values());boot={};checks={}
if completed:
 samples=[]
 for group in ['W0','W1','W2']:
  path=E/'validation_per_sample'/group/'epoch-110.csv.gz'
  with gzip.open(path,'rt') as f:rows=list(csv.DictReader(f))
  assert len(rows)==4500
  samples.extend(rows)
 write('validation_per_sample.csv',samples);by={g:{int(x['sample_id']):x for x in samples if x['group']==g} for g in final};rng=np.random.default_rng(20261128)
 for ref in ['W0','W2']:
  ids=sorted(by['W1']);assert ids==sorted(by[ref]) and all(by['W1'][i]['label']==by[ref][i]['label'] for i in ids);a=np.array([int(by['W1'][i]['correct'])-int(by[ref][i]['correct']) for i in ids]);d=np.array([float(by['W1'][i]['loss'])-float(by[ref][i]['loss']) for i in ids]);ix=rng.integers(0,len(ids),size=(2000,len(ids)));boot['W1-'+ref]={'accuracy_delta_pp':100*float(a.mean()),'accuracy_ci95_pp':(100*np.quantile(a[ix].mean(1),[.025,.975])).tolist(),'CE_delta':float(d.mean()),'CE_ci95':np.quantile(d[ix].mean(1),[.025,.975]).tolist(),'per_sample_sop':'NA_not_collected'}
 dump('paired_bootstrap.json',{'seed':20261128,'rng':'numpy.default_rng PCG64','resamples':2000,'method':'percentile','comparisons':boot});write('validation_summary.csv',list(final.values()))
 for ref in ['W0','W2']:
  a=final['W1'];b=final[ref];acc=float(a['validation_accuracy'])-float(b['validation_accuracy']);sop=float(a['SOP_sample_weighted'])/float(b['SOP_sample_weighted']);checks[ref]={'SOP_relative':sop,'accuracy_delta_pp':100*acc,'path_A':sop<=.95 and acc>=-.003,'path_B':sop<=1.01 and acc>=.003}
 pathA=all(x['path_A'] for x in checks.values());pathB=all(x['path_B'] for x in checks.values());dens=all(float(final[g]['min_penalized_density'])>=.01 for g in final);gate.update(H023_path_A=pathA,H023_path_B=pathB,min_density_gate=dens,progression_passed=(pathA or pathB) and dens,comparisons=checks);dump('hypothesis_gate_results.json',gate)
 plt.figure(figsize=(6,4))
 for g,x in final.items():plt.scatter(float(x['SOP_sample_weighted'])/1e6,100*float(x['validation_accuracy']),label=g)
 plt.xlabel('SOP/sample (million)');plt.ylabel('accuracy (%)');plt.legend();save('endpoint_tradeoff.png')
else:dump('paired_bootstrap.json',{'status':'NA_incomplete_groups','reason':'fixed epoch 110 not available for all groups'});gate.update(H023_path_A=None,H023_path_B=None,progression_passed=None,reason='incomplete epoch110 groups');dump('hypothesis_gate_results.json',gate)
write('resource_usage.csv',[{'run_id':s.get('run_id','EXP-023-RUN-001'),'gpu_uuid':s.get('gpu_uuid'),'wall_seconds':s.get('wall_seconds'),'cpu_seconds':s.get('cpu_seconds'),'gpu_limit_seconds':2700,'cpu_limit_seconds':5400,'peak_allocated_bytes':s.get('peak_allocated_bytes'),'peak_reserved_bytes':s.get('peak_reserved_bytes'),'note':'conservative run wall; reporting/imports/short-lived children may be omitted'}])
lines=[]
for g in final:
 x=final[g]
 if x:lines.append(f"| {g} | {x['epoch']} | {float(x['validation_accuracy'])*100:.5f}% | {float(x['validation_loss']):.8f} | {float(x['FR_sample_weighted']):.8f} | {float(x['SOP_sample_weighted']):,.2f} | {float(x['all_sparse_weight_sparsity'])*100:.5f}% |")
 else:lines.append(f"| {g} | {s.get('completed',{}).get(g,'NA')} | NA | NA | NA | NA | NA |")
exp=[]
for x in ex:
 if x['epoch']=='100':exp.append(f"| {x['dataset']} | {x['layer']} | {float(x['p']):.8f} | {float(x['SOP_R94']):,.3f} | {float(x['normalized_error']):.2g} |")
coef=[]
for x in co:coef.append(f"| {x['group']} | {x['layer']} | {float(x['original_K']):.3f} | {float(x['Khat']):.6f} |")
(E/'REPORT.md').write_text(f'''# EXP-023：脉冲暴露感知权重稀疏训练

执行状态：{s.get('status')}；代码 `{s.get('code_sha','NA')}`；停止原因/异常：{s.get('error',s.get('stop_reason','无'))}。父检查点 epoch100 SHA256 `{js('parent_checkpoint_audit.json',{}).get('expected','NA')}`，只读保留。EXP-020/021/022原负面证据保留，MP-EFR/TP-EFR主线停止。

## H-022 暴露诊断

预设三个K=1024层的D_grad与D_cal比较：`{json.dumps(gate.get('details',{}),ensure_ascii=False)}`；进入训练门槛：{gate.get('H022_direction_gate_passed','NA')}。实际全层数值：

| 数据 | 层 | p | 重建 SOP | 相对误差 |
|---|---|---:|---:|---:|
{chr(10).join(exp) if exp else '| NA | NA | NA | NA | NA |'}

CPU显式整数枚举及真实二值输入下R94与作者监控器检查见 exposure_formula_tests.json、sop_reconstruction_audit.json；实际稀疏权重和BN行为审查见 baseline_formula_audit.md。D_grad/D_cal曾参与父模型训练。

## 固定系数

| 组 | 层 | 原K | Khat |
|---|---|---:|---:|
{chr(10).join(coef) if coef else '| NA | NA | NA | NA |'}

W2为固定随机置换的暴露对应关系对照，K值不同意味着其Khat系数分布不一定和W1相同。固定系数及进训练前验证见 selection_lock.json、preflight_tests.json；缺失即该阶段未执行。

## 固定第110轮评价

| 组 | 完成轮 | Accuracy | CE | FR | SOP/sample | 权重稀疏率 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

仅三组均完成第110轮才能作主要比较。推进路径和最低密度结果：`{json.dumps(checks,ensure_ascii=False)}`；完整门槛：`{json.dumps(gate,ensure_ascii=False)}`。配对Bootstrap见 paired_bootstrap.json；如某组未完成则为NA。各层稀疏率、阈值累计误差、连接剪除与重新激活见 layer_sparsity.csv、threshold_increment_audit.csv、topology_transition.csv（如已生成），均基于真实记录。

## 资源和限制

累计保守运行墙钟 {s.get('wall_seconds','NA')} 秒，CPU计数 {s.get('cpu_seconds','NA')} 秒；使用单卡 `{s.get('gpu_uuid','NA')}`。CPU报告生成、导入和极短生命周期子进程可能未完全计入；理论SOP不代表实测能耗或速度。单父训练种子，训练来源诊断和反复用于开发的验证集不是独立确认。R93–R95为固定活动下计数，不证明训练因果或原创性。缺失的正式训练及图表均标记NA，不以诊断代替结果。
''')
(E/'limitations.md').write_text('单训练种子；暴露样本参与父模型训练，验证集已多轮用于开发；理论SOP非硬件能耗；W2按暴露置换而非系数分布严格匹配；CPU统计可能漏计导入及短生命周期子进程。\n')
p=ROOT/'research/EXPERIMENT_REGISTRY.csv';f=p.open();reader=csv.DictReader(f);fields=reader.fieldnames;rr=list(reader);f.close()
for x in rr:
 if x['run_id']=='EXP-023-RUN-001':x.update(code_sha=s.get('code_sha',''),execution_status='COMPLETED' if s.get('status')=='COMPLETED' else 'PARTIAL',start_time_utc=s.get('start_utc',''),end_time_utc=s.get('end_utc',''),wall_time_seconds=str(s.get('wall_seconds','')),notes='H022 gate '+str(gate.get('H022_direction_gate_passed','NA'))+'; completed '+str(s.get('completed',{}))+'; H023 gate '+str(gate.get('progression_passed','NA')))
with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rr)
with (ROOT/'research/RESEARCH_STATE.md').open('a') as f:f.write('\nROUND-021 EXP-023 delivery: status '+str(s.get('status'))+', completed '+str(s.get('completed',{}))+'; H-022 admission '+str(gate.get('H022_direction_gate_passed','NA'))+', H-023 advancement '+str(gate.get('progression_passed','NA'))+'. H-022/H-023 remain UNTESTED pending theory review; prior negative evidence retained. See ../experiments/EXP-023/REPORT.md.\n')
dump('delivery_resources.json',{'cpu_process_seconds':time.process_time()-begin,'gpu_used':False});print('finalization done',s.get('status'),flush=True)
