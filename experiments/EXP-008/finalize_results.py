import json,csv,hashlib,os,datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/hym/keyan1');O=R/'experiments/EXP-008';m=json.loads((O/'metrics.json').read_text());rows=m['summary'];groups={r['group']:r for r in rows};fig=O/'figures';fig.mkdir(exist_ok=True)
def save(name):plt.tight_layout();plt.savefig(fig/(name+'.png'));plt.close()
plt.figure()
for x in rows:plt.scatter(x['SOPs']/1e6,x['accuracy']);plt.annotate(x['group'],(x['SOPs']/1e6,x['accuracy']))
plt.xlabel('SOPs per sample (millions)');plt.ylabel('validation accuracy');save('accuracy_vs_sops')
epochs=list(csv.DictReader((O/'epoch_metrics.csv').open()))
plt.figure()
for g in ['F0','F1','F2','M2']:
 a=[r for r in epochs if r['group']==g];plt.plot([int(r['epoch']) for r in a],[float(r['train_total_loss']) for r in a],label=g)
plt.xlabel('Epoch');plt.ylabel('train total loss');plt.legend();save('train_loss')
plt.figure();plt.bar(groups.keys(),[r['SOPs']/1e6 for r in rows]);plt.ylabel('SOPs/sample (million)');save('sops')
plt.figure();plt.bar(groups.keys(),[r['accuracy'] for r in rows]);plt.ylabel('validation accuracy');save('accuracy')
layers=list(csv.DictReader((O/'layer_metrics.csv').open()));plt.figure(figsize=(13,4))
for g in groups:
 a=[r for r in layers if r['group']==g];plt.plot(range(len(a)),[float(r['FR']) for r in a],label=g)
plt.xticks(range(19),[r['module'] for r in layers if r['group']=='B0'],rotation=90,fontsize=6);plt.ylabel('FR');plt.legend();save('layer_fr')
pre=json.loads((O/'runs/PREFLIGHT-001/smoke.json').read_text());cs=pre['checks']['counter_cases'];plt.figure();plt.bar(range(len(cs)),[x['relative_error'] for x in cs]);plt.axhline(1e-5,color='red');plt.yscale('log');plt.ylabel('proxy vs original relative error');save('sop_proxy_error')
plt.figure()
for g in ['F0','F1','F2','M2']:plt.scatter(groups[g]['sop_savings']*100,groups[g]['accuracy']*100,label=g)
plt.xlabel('SOP savings (%)');plt.ylabel('accuracy (%)');plt.legend();save('ablation')
lines=['# EXP-008: 预算约束微调与自蒸馏','',f"状态：{m['status']}；固定第5轮主结果。实际代码 `93ad0a558570458b2ca9066ac127a13acd5a435c`。",'', '仅官方训练分区中固定45000张训练、4500张验证；500张训练来源校准集只用于前置计数校验；诊断池和官方测试集未用。基线模型为第194轮检查点，四组相同初始化、顺序、增强和优化器，各5轮；阈值固定×1.25、禁止手动更新。','', '| 组别 | 验证准确率 | 交叉熵 | SOPs/样本 | SOP降幅 | FR_global | FR_author | Epoch | 成功更新 | AMP跳步 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['group']} | {r['accuracy']:.6f} | {r['loss']:.6f} | {r['SOPs']:.3f} | {r['sop_savings']:.6f} | {r['FR_global']:.6f} | {r['FR_author']:.6f} | {r.get('epochs',0)} | {r.get('successful_steps','—')} | {r.get('amp_skips','—')} |")
lines+=['','## 预设判断与配对Bootstrap',f"H-007效果门槛：{m['H007_gate']}；F0相对P0 SOP回升/基线={m['rebound_F0']:.6f}；F1对应值={m['rebound_F1']:.6f}。",f"H-008效果门槛：{m['H008_gate']}；M2与F1是否预算匹配：{m['screening_gate']['M2_F1_budget_match']}。",f"总体筛选信号：{m['screening_signal']}；各条件：`{json.dumps(m['screening_gate'],ensure_ascii=False)}`。",'']
for name,c in m['paired_bootstrap'].items():lines.append(f"- {name}: 正确率差 {c['correct']['delta']:.6f}，95% CI {c['correct']['ci95']}；交叉熵差 {c['loss']['delta']:.6f}，95% CI {c['loss']['ci95']}。")
lines+=['','## 工程核查与适用范围',f"代理与原始计数器8个条件的最大绝对误差 {max(x['absolute_error'] for x in cs):.6f} SOPs/样本，最大相对误差 {max(x['relative_error'] for x in cs):.9g}；阈值×1/×1.25及完整/不完整batch均覆盖。锁定C0=54945117.504，实测={pre['checks']['calibration_sops']['1.0']}，相对差={pre['checks']['C0_relative_difference']:.9g}，未修改锁定值。两个shortcut已包含；人工卷积扇出各项误差0。",'正阈值和实际连接来源下，代理前向计数与原Monitor等价；反向为原替代梯度而非硬阈值有限差分。超预算非零有限梯度、预算内hinge零梯度、教师冻结与学生阈值锁定通过。原始Monitor不计首层输入卷积和非卷积运算，也不代表实测能耗。','',f"内部计时：含两次前置检查墙钟 {m['wall_seconds_including_preflight']:.2f} 秒，进程及已结束数据加载子进程累计CPU {m['cpu_seconds_including_preflight']:.2f} 秒；Python启动和外部少量整理工作未纳入该内部计时。",'','逐样本配对数据、训练曲线、每层FR、真实批次SOP、检查点清单和运行日志均与本报告同目录。样本级区间只描述单模型固定验证集，不是跨训练种子区间；验证集曾用于上游检查点选择。H-007/H-008正式状态维持UNTESTED待理论研究员审查。','', '## 检查点']
for r in m['checkpoint_manifest']:lines.append(f"- {r['group']}: `{r['path']}`；{r['bytes']} bytes；SHA256 `{r['sha256']}`")
(O/'REPORT.md').write_text('\n'.join(lines)+'\n');(O/'limitations.md').write_text('单模型与固定验证集；无跨训练种子不确定性。验证集曾用于原检查点选择。SOPs为指定监控范围的理论计数，不是硬件能耗；没有逐样本SOP置信区间。\n')
state=R/'research/RESEARCH_STATE.md';state.write_text(state.read_text()+f"\nEXP-008 COMPLETED；H-007预设效果门槛={m['H007_gate']}，H-008预设效果门槛={m['H008_gate']}，整体筛选信号={m['screening_signal']}。正式科学假设状态仍UNTESTED待理论研究员审查；详情见../experiments/EXP-008/REPORT.md。\n")
reg=R/'research/EXPERIMENT_REGISTRY.csv'
with reg.open('a',newline='') as f:
 w=csv.writer(f)
 for g in ['B0','P0','F0','F1','F2','M2']:
  w.writerow(['EXP-008','EXP-008-'+g+'-RUN-001','ROUND-009','budget-distillation','H-007;H-008','v0.1.5','research/round-009-budget-distillation','93ad0a558570458b2ca9066ac127a13acd5a435c','',2030,'COMPLETED','UNTESTED','experiments/EXP-008/config.yaml','experiments/EXP-008/summary.csv','experiments/EXP-008/REPORT.md','','','','fixed five epochs for fine-tuning; B0/P0 inference'])
print(json.dumps({'report':str(O/'REPORT.md'),'status':m['status'],'H007':m['H007_gate'],'H008':m['H008_gate'],'screen':m['screening_signal']}))
