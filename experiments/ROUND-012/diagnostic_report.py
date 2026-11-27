import csv,json,shutil,sys,platform
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/hym/keyan1');O=R/'experiments/EXP-012';P=O/'runs/EXP-012-RUN-001';gate=json.load((P/'intervention_gate.json').open());audit=json.load((P/'independent_gate_audit.json').open());status=json.load((P/'status.json').open());rows=list(csv.DictReader((P/'threshold_step_diagnostics.csv').open()));cmp=list(csv.DictReader((P/'short_trajectory_comparison.csv').open()));fig=O/'figures';fig.mkdir(exist_ok=True)
def save(n):plt.tight_layout();plt.savefig(fig/(n+'.png'));plt.close()
plt.figure(figsize=(12,4))
for g in ['D0','G0-S','G1-S','M4-S']:
 z=[x for x in rows if x['group']==g and x['attempt']=='1'];plt.plot(range(len(z)),[abs(float(x['scaled_task_gradient'])) for x in z],label=g)
plt.yscale('log');plt.xticks(range(len(z)),[x['module'] for x in z],rotation=90,fontsize=5);plt.ylabel('|task threshold gradient|');plt.legend();save('mean_vs_sum_gradient')
plt.figure(figsize=(12,4));z=[x for x in rows if x['group']=='M4-S' and x['attempt']=='1'];plt.plot(range(len(z)),[max(float(x['ulp_up']),float(x['ulp_down'])) for x in z],label='local ULP');plt.plot(range(len(z)),[abs(float(x['optimizer_delta'])) for x in z],label='|optimizer delta|');plt.yscale('log');plt.xticks(range(len(z)),[x['module'] for x in z],rotation=90,fontsize=5);plt.legend();save('ulp_vs_update')
plt.figure()
for g in ['D0','G0-S','G1-S','M4-S']:
 z=[x for x in rows if x['group']==g];initial={x['module']:float(x['threshold_before']) for x in z if x['attempt']=='1'};points=[]
 for t in range(1,17):
  a=[x for x in z if int(x['attempt'])==t];points.append(np.mean([abs(float(x['threshold_after'])-initial[x['module']]) for x in a]))
 plt.plot(range(1,17),points,label=g)
plt.xlabel('batch attempt');plt.ylabel('mean absolute threshold displacement');plt.legend();save('threshold_trajectory')
plt.figure(figsize=(12,4));z=[x for x in rows if x['group']=='M4-S'];names=list(dict.fromkeys(x['module'] for x in z))
for key in ['optimizer_delta','manual_delta']:plt.plot(range(len(names)),[sum(abs(float(x[key])) for x in z if x['module']==n) for n in names],label=key)
plt.yscale('log');plt.xticks(range(len(names)),names,rotation=90,fontsize=5);plt.ylabel('cumulative absolute update');plt.legend();save('task_vs_manual_updates')
for n in ['threshold_step_diagnostics.csv','short_trajectory_comparison.csv','intervention_gate.json','independent_gate_audit.json']:shutil.copy2(P/n,O/n)
(O/'code_version.json').write_text(json.dumps({'EXP-012_code_sha':status['code_sha'],'ROUND-011_result_sha':'1fdace017f755a1504e55ee0ec493c6502ae00dd'},indent=2)+'\n');(O/'environment.txt').write_text(f'platform: {platform.platform()}\npython: {sys.version}\ntorch: {torch.__version__}\nCUDA build: {torch.version.cuda}\nGPU UUID: {status["gpu_uuid"]}\n')
lines=['# EXP-012 共享阈值有效干预 — COMPLETED','',f"代码 `{status['code_sha']}`；固定原第194轮检查点和原训练样本，4组相同初始化、相同16批增强数据。短程诊断不使用验证集或官方测试集。",'', '|组别|尝试|成功更新|AMP跳步|实际非零优化器阈值更新模块|','|---|---:|---:|---:|---:|']
for g in ['D0','G0-S','G1-S','M4-S']:
 x=gate['groups'][g];lines.append(f"| {g} | {x['attempts']} | {x['successful_updates']} | {x['AMP_skips']} | {x['nonzero_optimizer_update_modules']}/19 |")
lines+=['',f"H-012工程进入门槛：**{gate['H012_intervention_gate']}**；M4-S相对G0-S、G1-S分别有 {gate['comparisons']['G0-S']['modules_greater_than_8_ULP']} 和 {gate['comparisons']['G1-S']['modules_greater_than_8_ULP']} 层超过8个保守局部ULP。每组同一批输入哈希一致，所有阈值正且有限，无明显更新爆炸；局部ULP取两模型上下间隔的最大值。",'', '独立审计逐条复算1216个层级优化器更新及Nesterov动量，最大误差0 ULP；求和后缩放与记录的FP32 N×mean一致。各组阈值参数的weight decay均为0。', '', '完整batch128的N分别为4194304（7层）、2097152（6层）、1048576（6层）；尾batch72对应2359296、1179648、589824。每个N从神经元实际输入张量单时间步计算，不额外乘T。逐层映射见threshold_shapes.csv。', '', '30个独立人工case、8时间步、不同尺寸及不完整batch通过；float64最大归一化误差6.91e-16，float32最大3.96e-7。原AMP非有限梯度正确跳步，后缩放引入非有限值会在step前停止。真实FP32和AMP网络检查前向、脉冲、SOP与非阈值更新一致，成功优化器更新及手动更新按原次序记录。', '', f"含前置检查的内部墙钟 {status['wall_seconds_including_preflight']:.3f} 秒、CPU {status['cpu_seconds_including_preflight']:.3f} 秒；独立审计CPU {audit['cpu_seconds']:.3f} 秒另计，启动和绘图等少量时间未单独计量。非精确GPU设备独占时长。", '', '该门槛只验证工程干预有效，不代表分类收益。H-012和H-013的正式科学状态保留UNTESTED，待理论研究员审查。']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n');(O/'limitations.md').write_text('短程只覆盖同一模型、固定16个训练batch，工程非零更新和向量分离不代表性能收益或跨种子稳定性。实际训练使用FP32阈值和AMP，原手动更新策略在AMP跳步时仍执行。\n');(O/'status.json').write_text(json.dumps({'execution_status':'COMPLETED','run_id':'EXP-012-RUN-001','H-012':'UNTESTED','intervention_gate':gate['H012_intervention_gate']},indent=2)+'\n');print('EXP-012 report written')
