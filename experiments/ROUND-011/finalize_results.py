import csv,json,sys,platform,math
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/hym/keyan1');O=R/'experiments/EXP-011';m=json.loads((O/'metrics.json').read_text());st=json.loads((O/'status.json').read_text());audit=json.loads((O/'independent_audit.json').read_text());pre=json.loads((O/'real_network_gradient_check.json').read_text());mixed=json.loads((O/'preflight/mixed_precision.json').read_text());rows=m['summary'];by={r['tag']:r for r in rows}
def read(name):
 with (O/name).open(newline='') as f:return list(csv.DictReader(f))
epoch=read('epoch_metrics.csv');changes=read('threshold_final_changes.csv');grads=read('threshold_gradient_statistics.csv');updates=read('threshold_updates.csv');sel=json.loads((O/'selected_configs.json').read_text());fig=O/'figures';fig.mkdir(exist_ok=True)
def save(name):plt.tight_layout();plt.savefig(fig/(name+'.png'));plt.close()
r=np.linspace(-.5,3,400);g=np.maximum(1-np.abs(r-1),0);sp=(r>=1).astype(float);plt.figure();plt.plot(r,-g*g,label='G0 threshold factor');plt.plot(r,-g,label='G1 threshold factor');plt.plot(r,sp-r*g,label='M4 amplitude factor');plt.plot(r,-r*g,label='M4 reset factor (theta=1)');plt.legend(fontsize=7);plt.xlabel('m/theta');plt.ylabel('local threshold factor');save('gradient_factors')
fig0,ax=plt.subplots(1,2,figsize=(10,4))
for k in ['G0','G1','M4']:
 z=[x for x in epoch if x['group']==k];ax[0].plot([int(x['epoch']) for x in z],[float(x['train_loss']) for x in z],label=k);ax[1].plot([int(x['epoch']) for x in z],[float(x['train_accuracy']) for x in z],label=k)
ax[0].set_ylabel('training cross entropy');ax[1].set_ylabel('training accuracy');ax[0].legend();save('training_curves')
plt.figure(figsize=(12,4))
for k in ['G0','G1','M4']:
 z=[x for x in changes if x['group']==k];plt.plot(range(len(z)),[100*float(x['relative_change']) for x in z],label=k)
plt.xticks(range(len(z)),[x['module'] for x in z],rotation=90,fontsize=5);plt.ylabel('threshold change (%)');plt.legend();save('threshold_change')
plt.figure(figsize=(12,4));nams=[x['module'] for x in updates if x['group']=='M4' and x['epoch']=='5']
for key in ['optimizer_delta','manual_delta']:
 z=[x for x in updates if x['group']=='M4' and x['epoch']=='5'];plt.plot(range(len(z)),[abs(float(x[key])) for x in z],label=key)
plt.xticks(range(len(nams)),nams,rotation=90,fontsize=5);plt.ylabel('absolute threshold update');plt.legend();save('threshold_update_comparison')
plt.figure(figsize=(12,4));z=[x for x in grads if x['group']=='M4' and x['epoch']=='5']
for key in ['direct_mean','state_mean','reset_mean']:plt.plot(range(len(z)),[float(x[key]) for x in z],label=key)
plt.xticks(range(len(z)),[x['module'] for x in z],rotation=90,fontsize=5);plt.ylabel('signed threshold gradient component');plt.legend();save('m4_gradient_components')
lay=read('layer_metrics.csv');plt.figure(figsize=(12,4))
for tag in ['B0','G0-direct','G1-direct','M4-direct']:
 z=[x for x in lay if x['tag']==tag];plt.plot(range(len(z)),[float(x['FR']) for x in z],label=tag)
plt.xticks(range(len(z)),[x['module'] for x in z],rotation=90,fontsize=5);plt.ylabel('firing rate');plt.legend();save('layer_fr')
for suffix,tags in [('direct',['B0','G0-direct','G1-direct','M4-direct']),('budget',['C0-0.2','G0-0.2','G1-0.2','M4-0.2','C0-0.3','G0-0.3','G1-0.3','M4-0.3'])]:
 plt.figure()
 for tag in tags:
  x=by[tag];plt.scatter(x['SOPs']/1e6,x['accuracy']*100);plt.annotate(tag,(x['SOPs']/1e6,x['accuracy']*100),fontsize=7)
 plt.xlabel('SOPs/sample (million)');plt.ylabel('validation accuracy (%)');save('accuracy_sops_'+suffix)
# Full per-layer changes and per-step mechanism statistics stay in CSV.
lines=['# EXP-011 ARTG 阈值代理梯度 — COMPLETED','',f"运行代码 SHA `{m['code_sha']}`；G0 前四轮使用 `afe8596baccf5db172b9ae03cfaf3d59f198feca`，第五轮从第4轮检查点恢复；G1/M4 使用上述修复后代码。三组均完整5轮。",'', '固定45000训练样本、500训练来源校准样本和4500验证样本；原诊断池和官方测试集未用。固定第5轮模型，校准只在500训练来源样本上通过SOP筛选，锁定后再评价验证集。', '', '|组别|倍率|验证准确率|损失|SOPs/样本|相对B0降幅|','|---|---:|---:|---:|---:|---:|']
for x in rows:lines.append(f"| {x['tag']} | {x['alpha']:.6g} | {x['accuracy']*100:.4f}% | {x['loss']:.6f} | {x['SOPs']:.3f} | {x['savings_vs_B0']*100:.4f}% |")
lines+=['','## 筛选门槛及配对比较',f"H-011先导筛选={m['H011_screening_signal']}；条件 `{json.dumps(m['H011_conditions'])}`。正式假设状态仍为UNTESTED，待理论研究员审查。"]
for k,v in m['comparisons'].items():lines.append(f"- {k}: 准确率差 {v['correct']['delta']*100:.4f} 个百分点，95% CI {[round(x*100,6) for x in v['correct']['ci95']]}；损失差 {v['loss']['delta']:.6f}，95% CI {v['loss']['ci95']}；SOP降幅差 {v['budget_difference']*100:.4f} 个百分点，同预算={v['matched_budget']}。")
lines+=['','## 前置检查、修复和中断','人工梯度（float64/float32）、T=8显式STE独立参考图、真实网络前向、非阈值梯度与AMP成功/跳步均通过；真实网络固定batch上三组logits、二值脉冲、SOP和非阈值梯度一致。三次首次前置失败及其原始代码和数据保留在preflight-attempt-001至003。', '首次前置通过后G0开始训练，但随后发现M4在float16膜电位、float32大上游梯度下可能提前溢出。G0在第5轮中断，完成4轮，未训练G1/M4。修复后增加混合精度幅值/重置/双路径测试，重新完成全部前置测试；AMP真实网络非阈值梯度及logits一致。G0恢复自第4轮检查点，优化器、调度器、AMP scaler及所有随机状态恢复；中断前后的第5轮指定批次19层诊断逐字段相同。废弃的部分第5轮记录完整保存在interruption-001。','R37以与原实现一致的数值次序计算，数学上等价；R39的每时间步mean是协议规定的优化缩放，并非严格共享参数梯度sum。原模型前向与手动更新策略保留，包括AMP跳步时手动更新。','', '## 训练和阈值诊断']
for item in m['checkpoint_manifest']:
 g0=item['group'];z=[x for x in changes if x['group']==g0];ch=np.array([float(x['relative_change']) for x in z]);u=[x for x in updates if x['group']==g0];opt=np.array([abs(float(x['optimizer_delta'])) for x in u]);man=np.array([abs(float(x['manual_delta'])) for x in u]);gr=np.array([abs(float(x['task_gradient'])) if x['task_gradient'] not in ('nan','inf','-inf') else np.nan for x in u]);lines.append(f"- {g0}: 完成5轮，成功更新{item['successful_steps']}，AMP跳步{item['amp_skips']}；19层阈值相对变化范围 [{ch.min()*100:.6f}%, {ch.max()*100:.6f}%]；指定批次绝对优化器变化中位数 {np.median(opt):.9g}、手动变化中位数 {np.median(man):.9g}、任务阈值梯度绝对值中位数 {np.nanmedian(gr):.9g}。")
z=[x for x in grads if x['group']=='M4'];
for k in ['direct_mean','state_mean','reset_mean','direct_abs_mean','state_abs_mean','reset_abs_mean']:
 a=np.array([float(x[k]) for x in z]);lines.append(f"- M4 `{k}`：95个层×指定批次记录的均值 {a.mean():.9g}、最大绝对值 {np.abs(a).max():.9g}。")
lines+=['','各层、各批次项的有符号及绝对均值见threshold_gradient_statistics.csv，优化器和手动实际改变量见threshold_updates.csv；这些诊断仅覆盖每轮固定第9个batch。','', '## 资源、局限与交付',f"含首次运行、中断、前置测试及保守预留的墙钟统计 {m['wall_seconds_including_preflight']:.2f} 秒、CPU {m['cpu_seconds_including_preflight']:.2f} 秒；独立审计另用CPU {audit['cpu_seconds']:.2f} 秒，Python启动和作图等少量时间未精确计入。该值不是硬件实测能耗或精确GPU设备独占时长。单模型固定验证集，Bootstrap不是跨训练种子稳定性；验证集曾用于原检查点选择。",'','各最终检查点和SHA256：']
for x in m['checkpoint_manifest']:lines.append(f"- {x['group']}: `{x['path']}`；SHA256 `{x['sha256']}`；{x['bytes']} bytes。")
lines+=['','全部15个已完成轮次检查点、位置、大小和校验值见checkpoint_manifest_all.csv；权重保留服务器，未上传GitHub。']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n');(O/'limitations.md').write_text('单个原始训练种子与固定验证集，样本级CI不是跨种子稳定性。阈值诊断只覆盖每轮预设的一个batch。原Monitor计数不是硬件能耗。G0部分第五轮因AMP实现问题中断并从第四轮恢复，未使用废弃部分的结果；详细审计见interruption-001。\n');(O/'environment.txt').write_text(f"platform: {platform.platform()}\npython: {sys.version}\ntorch: {torch.__version__}\nCUDA build: {torch.version.cuda}\nGPU: {st['gpu_uuid']}\n")
(O/'code_version.json').write_text(json.dumps({'G0_epochs_1_to_4':'afe8596baccf5db172b9ae03cfaf3d59f198feca','G0_epoch_5_and_G1_M4':m['code_sha'],'prior_round_result_sha':'8202467fe7cc26e3b19bb82aa51a8f4e25e04410'},indent=2)+'\n')
state=R/'research/RESEARCH_STATE.md';state.write_text(state.read_text()+f"\nEXP-011 COMPLETED；H-011预设筛选={m['H011_screening_signal']}。G0前四轮后因发现M4混合精度问题中断并恢复，失败记录和两个代码版本均已保存；全部真实结果详见../experiments/EXP-011/REPORT.md。H-011正式状态仍UNTESTED，待理论研究员审查。\n")
reg=R/'research/EXPERIMENT_REGISTRY.csv'
with reg.open('a',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['EXP-011','EXP-011-G0-RUN-001','ROUND-011','threshold-gradient','H-011','v0.1.7','research/round-011-threshold-gradient','afe8596baccf5db172b9ae03cfaf3d59f198feca','',2030,'PARTIAL','UNTESTED','experiments/EXP-011/config.json','experiments/EXP-011/interruption-001/epoch_metrics.csv','experiments/EXP-011/REPORT.md','','','','stopped during epoch 5; checkpoint epoch4; completed via G0-RUN-002'])
 for x in m['checkpoint_manifest']:
  g=x['group'];runid='EXP-011-'+g+('-RUN-002' if g=='G0' else '-RUN-001');w.writerow(['EXP-011',runid,'ROUND-011','threshold-gradient','H-011','v0.1.7','research/round-011-threshold-gradient',m['code_sha'],'',2030,'COMPLETED','UNTESTED','experiments/EXP-011/config.json','experiments/EXP-011/summary.csv','experiments/EXP-011/REPORT.md','','',x['wall_seconds'],'G0 resumed epoch4 if applicable; fixed five epochs'])
print(json.dumps({'status':'COMPLETED','H011_screen':m['H011_screening_signal']}))
