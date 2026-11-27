import csv,json,sys,platform
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/hym/keyan1');O=R/'experiments/EXP-013';D=R/'experiments/EXP-012';m=json.load((O/'metrics.json').open());st=json.load((O/'status.json').open());audit=json.load((O/'independent_audit.json').open());diag=json.load((D/'intervention_gate.json').open());groups=['G0-S','G1-S','M4-S'];summ=m['summary'];by={r['tag']:r for r in summ}
def read(n):
 with (O/n).open(newline='') as f:return list(csv.DictReader(f))
epoch=read('epoch_metrics.csv');agg=read('threshold_update_summary.csv');detail=read('threshold_gradient_statistics.csv');change=read('threshold_final_changes.csv');fig=O/'figures';fig.mkdir(exist_ok=True)
def save(n):plt.tight_layout();plt.savefig(fig/(n+'.png'));plt.close()
fig1,ax=plt.subplots(1,2,figsize=(10,4))
for group in groups:
 z=[x for x in epoch if x['group']==group];ax[0].plot([int(x['epoch']) for x in z],[float(x['train_loss']) for x in z],label=group);ax[1].plot([int(x['epoch']) for x in z],[float(x['train_accuracy'])*100 for x in z],label=group)
ax[0].set_ylabel('training cross entropy');ax[1].set_ylabel('training accuracy (%)');ax[0].legend();save('training_curves')
plt.figure(figsize=(12,4))
for group in groups:
 z=[x for x in change if x['group']==group];plt.plot(range(len(z)),[100*float(x['relative_change']) for x in z],label=group)
plt.xticks(range(len(z)),[x['module'] for x in z],rotation=90,fontsize=5);plt.ylabel('threshold relative change (%)');plt.legend();save('threshold_changes')
plt.figure(figsize=(12,4));names=[x['module'] for x in agg if x['group']=='M4-S' and x['epoch']=='1']
for typ in ['optimizer_abs_cumulative','manual_abs_cumulative']:
 vals=[sum(float(x[typ]) for x in agg if x['group']=='M4-S' and x['module']==name) for name in names];plt.plot(range(len(names)),vals,label=typ)
plt.yscale('log');plt.xticks(range(len(names)),names,rotation=90,fontsize=5);plt.ylabel('five-epoch cumulative absolute delta');plt.legend();save('task_vs_manual_updates')
plt.figure(figsize=(12,4));z=[x for x in detail if x['group']=='M4-S' and x['epoch']=='5']
for key in ['direct_mean_sum_equivalent','state_mean_sum_equivalent','reset_mean_sum_equivalent']:plt.plot(range(len(z)),[float(x[key]) for x in z],label=key)
plt.xticks(range(len(z)),[x['module'] for x in z],rotation=90,fontsize=5);plt.ylabel('signed threshold-gradient components');plt.legend(fontsize=7);save('m4_gradient_components')
lay=read('layer_metrics.csv');plt.figure(figsize=(12,4))
for tag in ['B0','G0-S-direct','G1-S-direct','M4-S-direct']:
 z=[x for x in lay if x['tag']==tag];plt.plot(range(len(z)),[float(x['FR']) for x in z],label=tag)
plt.xticks(range(len(z)),[x['module'] for x in z],rotation=90,fontsize=5);plt.ylabel('firing rate');plt.legend();save('layer_fr')
for suff,tags in [('direct',['B0']+[g+'-direct' for g in groups]),('budget',[g+'-'+t for t in ['0.2','0.3'] for g in ['C0']+groups])]:
 plt.figure()
 for tag in tags:
  x=by[tag];plt.scatter(x['SOPs']/1e6,x['accuracy']*100);plt.annotate(tag,(x['SOPs']/1e6,x['accuracy']*100),fontsize=7)
 plt.xlabel('SOPs/sample (million)');plt.ylabel('validation accuracy (%)');save('accuracy_sops_'+suff)
lines=['# EXP-013 求和等价阈值梯度训练 — COMPLETED','',f"EXP-012所有预设工程进入条件及独立审计通过后，G0-S/G1-S/M4-S各自从原第194轮检查点重新初始化，阈值×1.25，各训练5轮。EXP-012执行代码 `{m['diagnostic_code_sha']}`；EXP-013执行代码 `{m['code_sha']}`。",'', '使用45000训练样本、固定500训练来源校准样本和4500验证样本，未使用科研诊断池或官方测试集。所有倍率和已学习层间阈值结构仅依据校准SOP选择并锁定后评价验证集。','', '|组别|倍率|准确率|损失|SOPs/样本|相对B0降幅|','|---|---:|---:|---:|---:|---:|']
for r in summ:lines.append(f"| {r['tag']} | {r['alpha']:.6g} | {r['accuracy']*100:.4f}% | {r['loss']:.6f} | {r['SOPs']:.3f} | {r['savings_vs_B0']*100:.4f}% |")
lines+=['','## 预设筛选与配对统计',f"H-013先导筛选 `{m['H013_screening_signal']}`；条件：`{json.dumps(m['H013_conditions'])}`。H-012/H-013正式科学状态保持UNTESTED，待理论研究员审查。"]
for name,r in m['comparisons'].items():lines.append(f"- {name}: 准确率差 {r['correct']['delta']*100:.4f}个百分点，95% CI {[round(x*100,6) for x in r['correct']['ci95']]}；损失差 {r['loss']['delta']:.6f}，95% CI {r['loss']['ci95']}；验证SOP降幅差 {r['budget_difference']*100:.4f}个百分点，同预算={r['matched_budget']}。")
lines+=['','## 全训练过程阈值更新与梯度诊断']
for info in m['checkpoint_manifest']:
 g=info['group'];a=[x for x in agg if x['group']==g];ch=np.array([float(x['relative_change']) for x in change if x['group']==g]);opt=sum(float(x['optimizer_abs_cumulative']) for x in a);manual=sum(float(x['manual_abs_cumulative']) for x in a);success=sum(int(x['successful_steps']) for x in a);zero=sum(float(x['optimizer_zero_proportion_success'])*int(x['successful_steps']) for x in a);pos=sum(int(x['optimizer_positive_count']) for x in a);neg=sum(int(x['optimizer_negative_count']) for x in a);lines.append(f"- {g}: 5轮，成功更新{info['successful_steps']}、AMP跳步{info['amp_skips']}；19层阈值相对初值变化 [{ch.min()*100:.4f}%, {ch.max()*100:.4f}%]；所有成功层步中实际优化器更新为零比例 {zero/success:.6f}，绝对优化器更新累计 / 手动更新累计 = {opt/manual if manual else float('nan'):.6g}，优化器更新正/负次数 {pos}/{neg}。")
lines.append('每层每轮的所有batch统计见threshold_update_summary.csv；该比例汇总了层×batch，不能作为跨模型统计。额外固定每轮第9个batch的逐层原始梯度、N、ULP、动量、实际更新及M4分量的放大前后结果见threshold_gradient_statistics.csv。')
for key in ['direct_mean','state_mean','reset_mean','direct_mean_sum_equivalent','state_mean_sum_equivalent','reset_mean_sum_equivalent']:
 z=[float(x[key]) for x in detail if x['group']=='M4-S' and x[key]!=''];lines.append(f"- M4-S `{key}`：95条层×指定batch记录的有符号均值 {np.mean(z):.9g}，绝对值最大 {np.max(np.abs(z)):.9g}。")
lines+=['',f"最终模型逐张量差异：`{json.dumps(audit['model_state_comparisons'])}`。全batch N在完整128样本和最后72样本均按真实输入计算，并由独立审计逐层核对；未重复乘T。原始AMP跳步后仍执行手动稀疏更新。",'', '## 实验完整性、资源与局限',f"数学聚合和AMP前置检查通过；EXP-012四组各16次成功更新，三个sum组各19/19层存在实际非零更新，M4-S对另两组各19/19层超过8个局部ULP。诊断独立逐条重算1216个层级优化器更新，最大误差0 ULP。",f"累计含前置检查和短程诊断、以及保守启动预留的墙钟 {st['wall_seconds_including_diagnostic']:.2f} 秒、CPU {st['cpu_seconds_including_diagnostic']:.2f} 秒；独立审计CPU {audit['cpu_seconds']:.2f} 秒另计。启动和制图等其他少量CPU时间未精确计入，不是GPU独占时间或硬件能耗。",'验证集曾用于上游检查点选择；样本级Bootstrap不反映独立训练种子的稳定性。没有根据验证结果调参或选择epoch。','', '最终检查点：']
for x in m['checkpoint_manifest']:lines.append(f"- {x['group']}: `{x['path']}`，SHA256 `{x['sha256']}`，{x['bytes']} bytes。")
lines+=['','全部15个已完成轮次检查点及哈希见checkpoint_manifest_all.csv，模型权重保留服务器。']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n');(O/'limitations.md').write_text('单训练种子与固定验证集；样本级CI不能解释为跨种子稳定性。原Monitor SOP不是实测能耗。梯度组件仅在每轮一个固定batch详记，所有batch的阈值实际更新按epoch汇总。\n');(O/'environment.txt').write_text(f'platform: {platform.platform()}\npython: {sys.version}\ntorch: {torch.__version__}\nCUDA build: {torch.version.cuda}\nGPU UUID: {st["gpu_uuid"]}\n');(O/'code_version.json').write_text(json.dumps({'EXP012_code_sha':m['diagnostic_code_sha'],'EXP013_code_sha':m['code_sha'],'ROUND011_result_sha':'1fdace017f755a1504e55ee0ec493c6502ae00dd'},indent=2)+'\n')
state=R/'research/RESEARCH_STATE.md';state.write_text(state.read_text()+f"\nEXP-012 COMPLETED，H-012工程进入门槛通过；EXP-013 COMPLETED，H-013预设筛选={m['H013_screening_signal']}。H-012/H-013正式科学状态仍为UNTESTED，待理论研究员审查；见../experiments/EXP-012/REPORT.md和../experiments/EXP-013/REPORT.md。\n")
reg=R/'research/EXPERIMENT_REGISTRY.csv'
with reg.open('a',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['EXP-012','EXP-012-RUN-001','ROUND-012','intervention-diagnostic','H-012','v0.1.8','research/round-012-effective-threshold-gradient',m['diagnostic_code_sha'],'',2030,'COMPLETED','UNTESTED','experiments/EXP-012/config.json','experiments/EXP-012/intervention_gate.json','experiments/EXP-012/REPORT.md','','','','16 successful updates per group; all gates passed'])
 for x in m['checkpoint_manifest']:
  g=x['group'];w.writerow(['EXP-013','EXP-013-'+g+'-RUN-001','ROUND-012','sum-gradient-training','H-013','v0.1.8','research/round-012-effective-threshold-gradient',m['code_sha'],'',2030,'COMPLETED','UNTESTED','experiments/EXP-013/config.json','experiments/EXP-013/summary.csv','experiments/EXP-013/REPORT.md','','',x['train_wall_seconds'],'EXP-012 gate passed; 5 epochs per group'])
print(json.dumps({'status':'COMPLETED','H013_screen':m['H013_screening_signal']}))
