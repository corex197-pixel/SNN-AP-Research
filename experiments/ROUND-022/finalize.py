"""CPU-only delivery with explicit NA when a gated stage did not execute."""
import os,time,json,csv,datetime
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];E=ROOT/'experiments/EXP-024';F=E/'figures';F.mkdir(exist_ok=True);start=time.process_time()
def read(n):
 p=E/n;return list(csv.DictReader(p.open())) if p.exists() else []
def js(n,default=None):
 p=E/n;return json.loads(p.read_text()) if p.exists() else default
def dump(n,v):(E/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def save(n):plt.tight_layout();plt.savefig(F/n,dpi=150);plt.close()
s=js('status.json',{'status':'PARTIAL','error':'status missing'});gate=js('hypothesis_gate_results.json',{});v=read('validation_summary.csv');inter=read('interaction_diagnostics.csv');stats=read('neuron_statistics.csv');trace=read('greedy_selection_trace.csv');boot=js('paired_bootstrap.json',{}).get('comparisons',{});admit=js('admission_gate_audit.json',{});struct=js('structural_equivalence_tests.json',{});actual=js('selected_structure_audit.json',{});parameter=js('parameter_count_audit.json',{})
if stats:
 plt.figure(figsize=(6,4));plt.scatter([float(x['spike_rate']) for x in stats],[float(x['single_delta_CE']) for x in stats],s=3,alpha=.5);plt.xlabel('neuron mean spike rate');plt.ylabel('exact single deletion ΔCE');save('rate_vs_single_loss.png')
if inter:
 fig,ax=plt.subplots(figsize=(6,4));idx=list(range(len(inter)));ax.bar([i-.18 for i in idx],[float(x['joint_delta_CE']) for x in inter],.36,label='actual joint');ax.bar([i+.18 for i in idx],[float(x['sum_single_deltas']) for x in inter],.36,label='sum singles');ax.set_xticks(idx,[x['group'] for x in inter]);ax.set_ylabel('ΔCE');ax.legend();save('interaction.png')
if trace:
 r=[x for x in trace if x['within_block_rank']=='1'];plt.figure(figsize=(6,4));plt.plot([int(x['round']) for x in r],[float(x['actual_joint_loss_after_block']) for x in r],marker='o');plt.xlabel('greedy block');plt.ylabel('joint CE');save('greedy_trace.png')
if v:
 names=[x['group'] for x in v];fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].bar(names,[int(x['parameter_count'])/1e6 for x in v]);ax[0].set_ylabel('parameters (million)');ax[1].bar(names,[float(x['SOP_sample_weighted'])/1e6 for x in v]);ax[1].set_ylabel('SOP/sample (million)');save('parameters_sops.png');fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].bar(names,[100*float(x['accuracy']) for x in v]);ax[0].set_ylabel('accuracy (%)');ax[1].bar(names,[float(x['CE']) for x in v]);ax[1].set_ylabel('CE');save('accuracy_CE.png')
if boot:
 plt.figure(figsize=(6,4));names=['M5-S1','M5-S2','M5-S3']
 for i,k in enumerate(names):
  if k in boot:
   x=boot[k];val=x['accuracy_delta_pp'];lo,hi=x['accuracy_ci95_pp'];plt.errorbar(val,i,xerr=[[val-lo],[hi-val]],fmt='o')
 plt.yticks(range(len(names)),names);plt.axvline(0,c='k',lw=.8);plt.xlabel('accuracy difference (percentage points)');save('paired_accuracy_ci.png')
rows=[]
for x in v:rows.append(f"| {x['group']} | {100*float(x['accuracy']):.5f}% | {float(x['CE']):.8f} | {float(x['FR_global']):.8f} | {float(x['SOP_sample_weighted']):,.2f} | {int(x['parameter_count']):,} | {100*float(x['compression_fraction']):.4f}% | {int(x['inference_file_bytes']):,} |")
pairs=[]
for k,x in boot.items():pairs.append(f"| {k} | {x['accuracy_delta_pp']:.5f} | [{x['accuracy_ci95_pp'][0]:.5f},{x['accuracy_ci95_pp'][1]:.5f}] | {x['CE_delta']:.8f} | [{x['CE_ci95'][0]:.6f},{x['CE_ci95'][1]:.6f}] |")
ints=[]
for x in inter:ints.append(f"| {x['group']} | {float(x['joint_delta_CE']):+.8f} | {float(x['sum_single_deltas']):+.8f} | {float(x['joint_minus_single_sum']):+.8f} |")
admission=admit.get('status','NA');tests=struct.get('fixed_cases',[]);maxerr=max([float(x['compact_error']) for x in tests],default=None);ae=js('exact_readout_tests.json',{}).get('R103_normalized_error','NA');gtest=gate.get('checks',{});lock=js('lock_commit.json',{}).get('sha','NA')
report=f'''# EXP-024：精确脉冲贡献与真实结构化剪枝

执行状态：{s.get('status','PARTIAL')}；实际代码 `{s.get('code_sha','NA')}`；锁定提交 `{lock}`；异常/停止原因 `{s.get('error','无')}`。U2第300轮为唯一父模型；历史EXP-023准入偏差及负面结果保持记录。

## 机器可核验准入与结构等价性

准入状态 `{admission}`；R103真实FP32对FP64代数重构归一化误差 `{ae}`；固定单个、16个及512个神经元的紧凑模型相对门控模型最大误差 `{maxerr if maxerr is not None else 'NA'}`。实际样本ID、分批[2,1]、断言、输出与退出状态见 admission_gate_audit.json；各锁定集合在D_cal/验证之前的实际等价性见 selected_structure_audit.json。未通过或未执行的测试不可视为通过。

## 验证结果

| 组 | Accuracy | CE | FR | SOP/sample | 实际参数量 | 结构压缩率 | 纯推理文件字节 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows) if rows else '| NA | NA | NA | NA | NA | NA | NA | NA |'}

各层SOP、FR、有效非零权重、稀疏率及时间见 layer_sops.csv、layer_FR.csv、validation_summary.csv；模型路径与SHA256见checkpoint_manifest.json。SOP为理论计数，不代表实测能耗或普适GPU加速。

## 联合损失与配对比较

| 组 | 联合ΔCE | 单项ΔCE之和 | 联合减单项和 |
|---|---:|---:|---:|
{chr(10).join(ints) if ints else '| NA | NA | NA | NA |'}

| 比较 | Accuracy差（百分点） | 95% CI | CE差 | 95% CI |
|---|---:|---:|---:|---:|
{chr(10).join(pairs) if pairs else '| NA | NA | NA | NA | NA |'}

配对Bootstrap采用PCG64、seed=20261128、2000次百分位区间；逐样本SOP未采集，其区间为NA。M5的分块内边际分数都基于同一块开始状态；联合CE重新计算，非单项和。

## 预注册门槛与限制

`{json.dumps(gtest,ensure_ascii=False)}`；整体推进 `{gate.get('progression_passed','NA')}`。H-024/H-025科学状态等待理论研究员审查；不可从公式正确推断性能有效。D_grad/D_cal曾参与父模型训练，验证集多轮用于开发，单种子结果非独立确认。贪心、线性读出及直接损失计分不作学术首创声明；该分解同样适用于满足条件的ANN。

单卡 `{s.get('gpu_uuid','NA')}`；保守预算墙钟 `{s.get('wall_seconds','NA')}` 秒，CPU计数 `{s.get('cpu_seconds','NA')}` 秒；导入、CPU汇总及短生命周期子进程可能漏计。未执行阶段按NA处理，严禁用合成值填补正式结果。
'''
(E/'REPORT.md').write_text(report);(E/'limitations.md').write_text('单训练种子；D_grad/D_cal参与训练，验证集已用于开发；固定25%及block16非全局最优；FP64离线贡献限特定线性读出；逐样本SOP未采集；理论SOP非真实能耗；CPU计数可能漏短子进程。\n')
with (E/'resource_usage.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['run_id','gpu_uuid','wall_seconds','cpu_seconds','peak_allocated_bytes','peak_reserved_bytes','gpu_limit_seconds','cpu_limit_seconds'],lineterminator='\n');w.writeheader();w.writerow({k:s.get(k,'NA') for k in ['run_id','gpu_uuid','wall_seconds','cpu_seconds','peak_allocated_bytes','peak_reserved_bytes']}|{'gpu_limit_seconds':1200,'cpu_limit_seconds':2700})
p=ROOT/'research/EXPERIMENT_REGISTRY.csv';f=p.open();reader=csv.DictReader(f);fields=reader.fieldnames;rr=list(reader);f.close()
for x in rr:
 if x['run_id']==s.get('run_id'):x.update(code_sha=s.get('code_sha',''),execution_status='COMPLETED' if s.get('status')=='COMPLETED' else 'PARTIAL',start_time_utc=datetime.datetime.fromtimestamp(s['start_unix'],datetime.timezone.utc).isoformat() if 'start_unix' in s else '',end_time_utc=datetime.datetime.fromtimestamp(s['end_unix'],datetime.timezone.utc).isoformat() if 'end_unix' in s else '',wall_time_seconds=str(s.get('wall_seconds','')),notes='admission '+str(admission)+'; progression '+str(gate.get('progression_passed','NA')))
with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rr)
with (ROOT/'research/RESEARCH_STATE.md').open('a') as f:f.write('\nROUND-022 EXP-024 delivery: '+str(s.get('status'))+'; admission '+str(admission)+'; progression '+str(gate.get('progression_passed','NA'))+'. H-024/H-025 UNTESTED pending researcher review. EXP-023 admission deviation remains preserved. See ../experiments/EXP-024/REPORT.md.\n')
dump('delivery_resources.json',{'cpu_process_seconds':time.process_time()-start,'gpu_used':False});print('delivery finalized',s.get('status'),flush=True)
