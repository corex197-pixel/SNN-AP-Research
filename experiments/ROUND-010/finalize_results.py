import csv,json,shutil,platform,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
R=Path('/home/hym/keyan1');O=R/'experiments/ROUND-010';D9=R/'experiments/EXP-009';D10=R/'experiments/EXP-010';A=D9/'runs/EXP-009-RUN-001';B=D10/'runs/EXP-010-RUN-001';m=json.loads((O/'metrics.json').read_text());status=json.loads((O/'status.json').read_text());audit=json.loads((O/'independent_audit.json').read_text());mode=list(csv.DictReader((B/'bn_mode_comparison.csv').open()));repeat=json.loads((B/'mode_repeat.json').read_text());summ=m['summary'];by={r['tag']:r for r in summ};comps=m['paired_bootstrap'];code=m['code_sha'];pre=json.loads((O/'preflight/results.json').read_text());costs=m['costs'];cal=json.loads((O/'calibration_baseline.json').read_text())
def writecsv(path,rows):
 if not rows:return
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def readcsv(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def savejson(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
# Keep raw shared outputs plus relevant copies in each experiment directory.
for root,run,exp,methods in [(D9,A,'EXP-009',{'C0','C1','C2'}),(D10,B,'EXP-010',{'C0','C3','M3'})]:
 for filename in ['validation_per_sample.csv','validation_batch_sops.csv','layer_metrics.csv']:
  rows=readcsv(O/filename);out=[z for z in rows if z['tag'] in ('B0','P0') or z['tag'].split('-')[0] in methods];writecsv(run/filename,out)
 writecsv(run/'summary.csv',[r for r in summ if r['group'] in methods|{'B0','P0'}]);
 savejson(run/'metrics.json',{'execution_status':m['status'],'summary':[r for r in summ if r['group'] in methods|{'B0','P0'}],'paired_bootstrap':{k:v for k,v in comps.items() if k.split('-')[0] in methods and k.split('_vs_')[1].split('-')[0] in methods},'H010_conditions':m['H010_conditions'] if exp=='EXP-010' else {},'H009_exploratory_gate':repeat['H009_exploratory_gate'] if exp=='EXP-010' else None,'costs':[x for x in costs if x['group'] in methods]})
 savejson(root/'status.json',{'execution_status':m['status'],'run_id':exp+'-RUN-001','scientific_status':'UNTESTED','code_sha':code,'partial':m['partial']});savejson(root/'code_version.json',{'execution_code_sha':code,'previous_round_result_sha':'1cf8ac6505e590b05d4c41a775054409700d11ab','baseline_sha256':audit['checkpoints_sha256']['baseline'],'F0_sha256':audit['checkpoints_sha256']['F0'],'F2_sha256':audit['checkpoints_sha256']['F2']});(root/'environment.txt').write_text(f'platform: {platform.platform()}\npython: {sys.version}\ntorch: {torch.__version__}\nCUDA build: {torch.version.cuda}\nGPU UUID: {status["gpu_uuid"]}\nGPU count: 1; FP32; no gradients; deterministic preprocessing\n');shutil.copy2(O/'commands.sh',root/'commands.sh');shutil.copy2(O/'run.log',run/'run.log');shutil.copy2(run/'summary.csv',root/'summary.csv');shutil.copy2(run/'metrics.json',root/'metrics.json')
(root/'limitations.md').write_text('单模型单验证集；验证集曾用于原检查点选择，样本级置信区间不能解释为跨种子稳定性。SOPs仅为原Monitor活动运算计数，不是硬件能耗。BN batch模式依赖当前batch同伴，不是常规逐样本部署评价。BN模式输出差异记录为每通道输出均值/方差差，不是逐元素绝对差。\n')
# R31 effective per-channel counts derived from verified execution shapes; dimensions unchanged by batch size.
records=[]
for name,shape in pre['checks']['BN_input_shapes'].items():records.append({'BN':name,'T':shape[0],'channels':shape[2],'height':shape[3],'width':shape[4],'calibration_samples':500,'per_channel_observation_count':shape[0]*500*shape[3]*shape[4]})
writecsv(B/'bn_population_counts.csv',records)
fig=D10/'figures';fig.mkdir(exist_ok=True);(D9/'figures').mkdir(exist_ok=True)
def save(n):plt.tight_layout();plt.savefig(fig/(n+'.png'));plt.close();shutil.copy2(fig/(n+'.png'),D9/'figures'/(n+'.png'))
plt.figure()
for z in summ:
 plt.scatter(z['SOPs']/1e6,z['accuracy']*100);plt.annotate(z['tag'],(z['SOPs']/1e6,z['accuracy']*100),fontsize=7)
plt.xlabel('SOPs/sample (million)');plt.ylabel('Validation accuracy (%)');save('accuracy_sops')
plt.figure()
for g in ['C0','C3','M3']:
 rr=[z for z in summ if z['group']==g];plt.plot([r['savings_vs_B0']*100 for r in rr],[r['accuracy']*100 for r in rr],marker='o',label=g)
plt.legend();plt.xlabel('SOPs savings (%)');plt.ylabel('Accuracy (%)');save('bn_calibration')
fig_,axs=plt.subplots(1,2,figsize=(10,4));labels=['C0','C1','C2','C3','M3'];v=[by[g+'-0.2'] for g in labels];axs[0].bar(labels,[x['accuracy']*100 for x in v]);axs[0].set_ylabel('Accuracy (%)');axs[1].bar(labels,[x['loss'] for x in v]);axs[1].set_ylabel('Cross entropy');save('matched_accuracy_loss')
base={(x['BN'],x['channel']):float(x['running_mean_before']) for x in readcsv(B/'bn_mode_channels.csv') if x['tag']=='B0-evalBN'};stats=readcsv(B/'bn_statistics.csv');plt.figure(figsize=(12,4))
for g in ['C3','M3']:
 alpha=json.loads((B/'selected_configs.json').read_text())[g]['0.2']['alpha'];d=defaultdict(list)
 for z in stats:
  if z['group']==g and abs(float(z['alpha'])-alpha)<1e-9:d[z['BN']].append(abs(float(z['running_mean'])-base[(z['BN'],z['channel'])]))
 names=list(d);plt.plot(range(len(names)),[np.mean(d[n]) for n in names],label=g)
plt.xticks(range(len(names)),names,rotation=90,fontsize=5);plt.ylabel('Mean absolute running-mean change');plt.legend();save('bn_statistics_change')
lay=readcsv(B/'bn_mode_layers.csv');plt.figure(figsize=(12,4))
for group in ['B0','P0','F0']:
 x=[z for z in lay if z['group']==group];plt.plot(range(len(x)),[float(z['delta_FR']) for z in x],label=group)
plt.xticks(range(len(x)),[z['module'] for z in x],rotation=90,fontsize=5);plt.ylabel('FR(batch BN) − FR(eval BN)');plt.legend();save('bn_mode_layer_fr')
plt.figure();plt.bar([x['group'] for x in costs],[x['seconds'] for x in costs]);plt.ylabel('Calibration search time (s)');save('calibration_cost')
# Per-experiment reports preserve real 20% and 30% results and paired uncertainty.
def table(methods):
 lines=['|目标|组别|倍率|准确率|交叉熵|SOPs/样本|相对B0降幅|','|---:|---|---:|---:|---:|---:|---:|']
 for target in ['0.2','0.3']:
  for g in methods:
   v=by[g+'-'+target];lines.append(f"| {float(target)*100:.0f}% | {g} | {v['alpha']:.5g} | {v['accuracy']*100:.4f}% | {v['loss']:.6f} | {v['SOPs']:.3f} | {v['savings_vs_B0']*100:.4f}% |")
 return '\n'.join(lines)
def comparison(keys):
 out=[]
 for k in keys:
  v=comps[k];a=v['correct'];b=v['loss'];out.append(f"- {k}：准确率差 {a['delta']*100:.4f} 个百分点，95% CI {np.array(a['ci95'])*100}；损失差 {b['delta']:.6f}，95% CI {b['ci95']}；验证SOP降幅差 {v['budget_difference']*100:.4f} 个百分点，匹配={v['matched_budget']}。")
 return '\n'.join(out)
common=f"固定194轮检查点及两个ROUND-009第五轮检查点，500张训练来源校准、4500张原验证，先锁定倍率与BN统计再评价。FP32、batch128、原SOPMonitor、无梯度训练，不用诊断池和官方测试集；代码SHA `{code}`。独立核查校验值、候选选择、配对样本与Bootstrap通过。B0/P0与EXP-008准确率、损失和SOP数值完全一致。校准基线C0_cal={cal['calibration_SOPs']}，较54945117.504差{cal['difference']:.3f}。"
(D9/'REPORT.md').write_text('# EXP-009 已有模型同预算评价 — COMPLETED\n\n'+common+'\n\n'+table(['C0','C1','C2'])+'\n\n'+comparison([f'{a}-{t}_vs_{b}-{t}' for t in ['0.2','0.3'] for a,b in [('C1','C0'),('C2','C0'),('C2','C1')]])+'\n\n20%目标下C1/C2均低于简单阈值校准C0，未得到微调模型的实用优势；30%目标下C2点估计高0.6个百分点，但配对CI跨零，属于单模型先导观察。选择仅使用校准SOP，与校准或验证准确率无关。校准样本属于训练数据，验证集用于上游模型选择；无跨种子稳定性结论。\n')
(D10/'REPORT.md').write_text('# EXP-010 BN模式及校准 — COMPLETED\n\n'+common+'\n\n'+table(['C0','C3','M3'])+'\n\n'+comparison([f'{a}-{t}_vs_{b}-{t}' for t in ['0.2','0.3'] for a,b in [('C3','C0'),('M3','C0'),('M3','C3')]])+'\n\n## BN模式对照\n\n'+ '\n'.join(f"- {z['group']}：eval SOP={float(z['C_evalBN']):.3f}；batch BN SOP={float(z['C_batchBN']):.3f}；D_mode={float(z['D_mode'])*100:.4f}%。" for z in mode)+f"\n\nP0重复测量SOP绝对差{repeat['absolute_difference']}；H-009探索性幅度门槛={repeat['H009_exploratory_gate']}。batch BN输出依赖同批样本，不是常规逐样本部署结果；各BN输入及输出统计、运行量、逐层FR和输出通道均值/方差差保存于CSV，未测量逐元素配对绝对差。\n\n## 校准与筛选\n\nC3累计平均采用四个batch等权及PyTorch每batch无偏方差，最后一批116张；M3逐前向顺序、float64 Chan合并并使用总体方差，输入形状T,B,C,H,W，各通道计数见bn_population_counts.csv。权重、BN affine和阈值仅采用固定协议赋值；候选从各自原始检查点独立生成。全部BN均值/方差及实际选用统计已保存，运行后原检查点SHA不变。\n\nH-010总体筛选门槛={m['H010_gate']}，条件明细：`{json.dumps(m['H010_conditions'])}`。20%目标下M3较C3的准确率差只有0.0222个百分点，配对CI跨零，不能宣称逐层校准独立必要。H-009/H-010正式状态均保持UNTESTED等待理论研究员审阅。\n")
for root in [D9,D10]:
 with (root/'REPORT.md').open('a') as f:f.write(f"\n## 资源、异常与限制\n\n包含两次前置检查的正式进程内部墙钟 {status['wall_seconds_including_preflight']:.3f} 秒、CPU {status['cpu_seconds_including_preflight']:.3f} 秒；独立CPU审计另外 {audit['cpu_seconds']:.3f} 秒，另有Python启动与绘图等未计入项，不能当作精确GPU设备占用。正式过程 {m['forwards_all']} 次完整前向，{m['samples_all']} 样本次；前置检查另 {pre['forwards']+2} 次前向。每组搜索耗时及前向次数见calibration_costs.json。首次后台启动未创建实际实验进程，第二次启动完成；未产生失败实验数据。无梯度训练或测试集访问。\n")
shutil.copy2(O/'calibration_costs.json',D9/'calibration_costs.json');shutil.copy2(O/'calibration_costs.json',D10/'calibration_costs.json')
state=R/'research/RESEARCH_STATE.md';state.write_text(state.read_text()+f"\nEXP-009、EXP-010 COMPLETED。20%目标下C1/C2不优于C0；P0 batch BN相对eval的SOP变化为 -27.4200%，且重复稳定，达到H-009预设探索性幅度。H-010筛选门槛未达到。H-009/H-010正式状态仍UNTESTED，待理论研究员审查；详见EXP-009/010报告。\n")
reg=R/'research/EXPERIMENT_REGISTRY.csv'
with reg.open('a',newline='') as f:
 w=csv.writer(f,lineterminator='\n')
 for exp,hy in [('EXP-009','H-007;H-008'),('EXP-010','H-009;H-010')]:w.writerow([exp,exp+'-RUN-001','ROUND-010','normalization-calibration',hy,'v0.1.6','research/round-010-normalization-calibration',code,'',2027,'COMPLETED','UNTESTED',f'experiments/{exp}/config.yaml',f'experiments/{exp}/metrics.json',f'experiments/{exp}/REPORT.md',status['start_utc'],status['end_utc'],status['wall_seconds_including_preflight'],'shared process; preflight counted; fixed calibration and validation'])
lit=R/'research/LITERATURE.md';s=lit.read_text().replace('Membrane Potential Batch Normalization for Spiking Neural Networks, ICCV 2023.','Membrane Potential Batch Normalization for Spiking Neural Networks, ICCV 2023, https://openaccess.thecvf.com/content/ICCV2023/html/Guo_Membrane_Potential_Batch_Normalization_for_Spiking_Neural_Networks_ICCV_2023_paper.html.').replace('Spiking Neural Networks Calibration, ICML 2021,','A Free Lunch From ANN: Towards Efficient, Accurate Spiking Neural Networks Calibration, ICML 2021,');lit.write_text(s)
print(json.dumps({'EXP009':'COMPLETED','EXP010':'COMPLETED','H009_gate':repeat['H009_exploratory_gate'],'H010_gate':m['H010_gate']}))
