"""Summarize actual EXP-017 artifacts only; safe for partial runs."""
import csv,json,os,hashlib,statistics
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parents[1]/'EXP-017';fig=O/'figures';fig.mkdir(exist_ok=True)
def read(name):
 p=O/name
 if not p.exists():return []
 with p.open(newline='') as f:return list(csv.DictReader(f))
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
r=read('epoch_metrics.csv');by={g:[x for x in r if x['group']==g] for g in ['U0','U1','U2']};status=json.loads((O/'status.json').read_text());lock=json.loads((O/'selection_lock.json').read_text());pre=json.loads((O/'preflight_status.json').read_text());sel=json.loads((O/'pilot_selection.json').read_text()) if (O/'pilot_selection.json').exists() else {'recommended':None,'reason':'Incomplete: required full 20-epoch groups unavailable'}
for fname,keys,title in [('accuracy.png',['train_accuracy','validation_accuracy'],'Accuracy'),('loss.png',['train_loss','validation_loss'],'Cross entropy'),('sparsity.png',['penalized_weight_sparsity'],'Penalized weight sparsity'),('firing_rate.png',['FR_sample_weighted'],'Sample-weighted firing rate'),('sops.png',['SOP_sample_weighted'],'Theoretical SOPs per sample')]:
 plt.figure(figsize=(8,4))
 for g,arr in by.items():
  for k in keys:
   if arr:plt.plot([int(x['epoch']) for x in arr],[float(x[k]) for x in arr],label=g+' '+k)
 plt.xlabel('Epoch');plt.ylabel(title);plt.legend(fontsize=7);plt.tight_layout();plt.savefig(fig/fname,dpi=130);plt.close()
for name,key,title in [('thresholds.png','threshold','Weight thresholds'),('densities.png','density','Layer effective density')]:
 rows=read('layer_sparsity.csv');plt.figure(figsize=(10,5))
 for g in by:
  for layer in sorted(set(x['layer'] for x in rows if x['group']==g)):
   a=[x for x in rows if x['group']==g and x['layer']==layer];plt.plot([int(x['epoch']) for x in a],[float(x[key]) for x in a],label=g+' '+layer)
 plt.xlabel('Epoch');plt.ylabel(title);plt.legend(fontsize=5,ncol=3);plt.tight_layout();plt.savefig(fig/name,dpi=130);plt.close()
fan=read('effective_fanout_statistics.csv');plt.figure(figsize=(9,4))
for g in by:
 a=[x for x in fan if x['group']==g and x['neuron']=='conv1.activation'];
 if a:plt.plot([int(x['epoch']) for x in a],[float(x['mean']) for x in a],marker='o',label=g+' effective')
if fan:plt.axhline(float(fan[0]['nominal']),color='black',ls='--',label='nominal (conv1.activation)')
plt.xlabel('Epoch');plt.ylabel('Mean effective fanout');plt.legend();plt.tight_layout();plt.savefig(fig/'fanout.png',dpi=130);plt.close()
plt.figure(figsize=(6,4))
for g,a in by.items():
 b=[x for x in a if int(x['epoch'])==20]
 if b:plt.scatter(float(b[0]['SOP_sample_weighted']),float(b[0]['validation_accuracy']),label=g)
plt.xlabel('SOP/sample');plt.ylabel('Accuracy');plt.legend();plt.tight_layout();plt.savefig(fig/'accuracy_vs_sops.png',dpi=130);plt.close()
manifest=O/'checkpoint_manifest.json';items=json.loads(manifest.read_text()) if manifest.exists() else [];known={x['path'] for x in items}
for g in by:
 for name in ['latest.pth','best_validation.pth','epoch-005.pth','epoch-010.pth','epoch-020.pth']:
  p=O/'runs/EXP-017-PILOT-001'/g/name
  if p.exists() and str(p) not in known:items.append({'group':g,'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p),'final_checkpoint_reload_verified':False})
manifest.write_text(json.dumps(items,indent=2)+'\n')
if not (O/'pilot_selection.json').exists():(O/'pilot_selection.json').write_text(json.dumps(sel,indent=2)+'\n')
lines=['# EXP-017 — UniSparse 受控训练工作点先导','',f"执行状态：{status['status']}。运行代码：`{status['code_sha']}`；前置检查：`{pre['code_sha']}`。H-016 限定 PARTIALLY_SUPPORTED；H-017 UNTESTED。",'',f"非零论文 λ 未确认；R68 λ_ref={lock['lambda_weight_neuron']['U1']:.12g}。U0=0，U1=λ_ref，U2=3λ_ref。训练与验证仅使用 CIFAR-10 训练来源固定划分，无官方测试。",'', '| 组 | 已完成轮数 | 第20轮准确率 | 第20轮损失 | 加权 FR | 加权 SOP/样本 | 受惩罚层稀疏率 | AMP 跳步 | 手动更新 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for g,a in by.items():
 z=[x for x in a if int(x['epoch'])==20];q=z[0] if z else None;v=lambda k:format(float(q[k]),'.8g') if q else 'NA';lines.append(f"| {g} | {len(a)} | {v('validation_accuracy')} | {v('validation_loss')} | {v('FR_sample_weighted')} | {v('SOP_sample_weighted')} | {v('penalized_weight_sparsity')} | {v('cumulative_amp_skips')} | {v('cumulative_manual_updates')} |")
lines+=['',f"预注册工作点选择：{json.dumps(sel,ensure_ascii=False)}。不以中途最佳轮次替代固定第20轮。",'',f"资源状态：{json.dumps(status,ensure_ascii=False)}。墙钟包含前置检查与等待，CPU 为进程及可见子进程计数估计；所有检查点和哈希见 checkpoint_manifest.json。",'', '原作者 SOP 将各批次计数平均后按配置 batch 归一；样本加权 SOP 使用总计数除以实际 4500 样本。尾批保留；SOP 为理论计数，非实测能耗。原始逐层阈值、密度、BN、有效扇出与逐样本数据均保存在本目录。','', '限制：单种子、同一验证集工作点筛选；20轮结果不能代表300轮稳定性。原始 AMP 跳步后手动惩罚语义保留。按轮采样与worker种子固定，检查点恢复了模型/优化器/调度器/scaler/RNG；跨环境、工作进程中断后的逐位可复现性未确认。CPU 统计可能遗漏进程启动及极短暂子进程。']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n');(O/'limitations.md').write_text(lines[-1]+'\n');print('report written',status['status'])
