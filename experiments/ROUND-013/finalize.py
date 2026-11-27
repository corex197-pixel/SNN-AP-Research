import os,sys,json,csv,hashlib,time,resource,copy
from pathlib import Path
import numpy as np,torch
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
start=time.monotonic();root=Path('/home/hym/keyan1');os.chdir(root);o=root/'experiments/EXP-014';(o/'figures').mkdir(exist_ok=True);sel=json.loads((o/'selected_channels.json').read_text());metrics=json.loads((o/'metrics.json').read_text());status=json.loads((o/'status.json').read_text());eq=json.loads((o/'structural_equivalence_tests.json').read_text());names=list(next(iter(sel.values())));sizes=[32,64,128]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ck=torch.load(root/'experiments/EXP-004/runs/EXP-004-RUN-003/checkpoint_best_validation.pth',map_location='cpu',weights_only=False);manifest=[];shapes=[]
for group,ss in sel.items():
 sd={n:v.detach().clone() for n,v in ck['model'].items()};changed=set()
 for n,size in zip(names,sizes):
  prefix=n.rsplit('.',1)[0];keep=[i for i in range(size) if i not in ss[n]]
  for suffix,dim in [('.0.weight',0),('.1.weight',0),('.1.bias',0),('.1.running_mean',0),('.1.running_var',0),('.3.weight',1)]:
   key=prefix+suffix;before=sd[key];ix=torch.tensor(keep,dtype=torch.long);sd[key]=torch.index_select(before,dim,ix);assert torch.equal(sd[key],torch.index_select(ck['model'][key],dim,ix));changed.add(key)
   if suffix in ['.0.weight','.3.weight']:shapes.append({'group':group,'module':n,'tensor':suffix,'original_shape':str(list(before.shape)),'compact_shape':str(list(sd[key].shape))})
 for n in sd:
  if n not in changed:assert torch.equal(sd[n],ck['model'][n]),n
 out=o/'runs/EXP-014-RUN-001'/f'{group}-compact.pth';torch.save({'model':sd,'selected_channels':ss,'source_checkpoint_sha256':'c068f54379cb5d29bcfa5fe2d9de676cd3cc153433ada1ed5aa1c66355d0b32a','run_code_sha':status['code_sha'],'note':'CPU reconstruction of evaluated compact tensors; all unsliced tensors bitwise equal to original'},out);manifest.append({'group':group,'path':str(out),'size_bytes':out.stat().st_size,'sha256':sha(out),'unchanged_tensors_bitwise_verified':True,'sliced_tensors_bitwise_verified':True})
(o/'checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with (o/'tensor_shapes.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=shapes[0],lineterminator="\n");w.writeheader();w.writerows(shapes)
def read(path):
 with path.open() as f:return list(csv.DictReader(f))
a=read(o/'channel_importance_A.csv');b=read(o/'channel_importance_B.csv');rows=read(o/'summary.csv');
def figsave(name):plt.tight_layout();plt.savefig(o/'figures'/name,dpi=140);plt.close()
for field,filename in [('FR','fr_vs_loss.png'),('L1','l1_vs_loss.png')]:
 fig,axs=plt.subplots(1,3,figsize=(13,4))
 for ax,n in zip(axs,names):
  rr=[r for r in a if r['module']==n];ax.scatter([float(r[field]) for r in rr],[float(r['D']) for r in rr],s=10);ax.set_title(n.split('.')[0]);ax.set_xlabel(field+' (A)');ax.set_ylabel('Single-channel ΔCE (A)');ax.axhline(0,color='gray',lw=.5)
 figsave(filename)
fig,axs=plt.subplots(1,3,figsize=(13,4))
for ax,n in zip(axs,names):
 aa=[r for r in a if r['module']==n];bb=[r for r in b if r['module']==n];ax.scatter([float(r['D']) for r in aa],[float(r['D']) for r in bb],s=10);ax.set_title(n.split('.')[0]);ax.set_xlabel('ΔCE A');ax.set_ylabel('ΔCE B')
figsave('loss_stability.png')
fig,axs=plt.subplots(1,3,figsize=(13,4));groups=[r['group'] for r in rows]
for ax,k,title in zip(axs,['parameters','accuracy','SOPs'],['Parameters','Validation accuracy','SOPs/sample']):ax.bar(groups,[float(r[k]) for r in rows]);ax.set_title(title);ax.tick_params(axis='x',rotation=45)
figsave('performance.png')
fig,axs=plt.subplots(3,1,figsize=(12,7))
for ax,n in zip(axs,names):
 for j,(g,ss) in enumerate(sel.items()):ax.scatter(ss[n],[j]*len(ss[n]),s=10)
 ax.set_yticks(range(4),list(sel));ax.set_title(n);ax.set_xlabel('Original channel index')
figsave('selected_channels.png')
fig,ax=plt.subplots(figsize=(10,4));positions=np.arange(3);orig=np.array(sizes);ax.bar(positions-.2,orig,.4,label='original');ax.bar(positions+.2,orig*.75,.4,label='compact');ax.set_xticks(positions,[n.split('.')[0] for n in names]);ax.set_ylabel('Conv1 output / Conv2 input channels');ax.legend();figsave('tensor_shapes.png')
stat=read(o/'ranking_statistics.csv');lines=['# EXP-014 — 结构化内部通道剪枝先导','',f"执行状态：{status['execution_status']}；运行代码：`{status['code_sha']}`。正式科研假设 H-014 / H-015 继续 UNTESTED，待理论研究员审查。",'', 'A、B 各 250 张，均来自训练来源样本，每组完成 224 个通道测量。A 锁定选择，B 仅用于稳定性统计；4500 验证样本在锁定后使用。R47—R50 与代码对应关系详见 [任务协议](TASK.md) 和 [运行代码](../ROUND-013/run.py)。', '', '## 验证结果', '', '| 组 | 准确率 | CE | 参数 | SOPs/样本 | SOP 相对 B0 节省 | FR_global |','|---|---:|---:|---:|---:|---:|---:|']
for r in rows:lines.append('| {} | {:.6f} | {:.6f} | {} | {:.2f} | {:.2%} | {:.6f} |'.format(r['group'],float(r['accuracy']),float(r['loss']),r['parameters'],float(r['SOPs']),float(r['SOP_savings']),float(r['FR_global'])))
lines+=['',f"H-014 预设先导门槛：{metrics['H014_screening_gate']}；H-015 预设先导门槛：{metrics['H015_screening_gate']}。所有剪枝组删除 96,880 个参数（8.93%），但没有一组的准确率下降不超过 1 个百分点。P-Loss 表现低于 P-FR 和 P-L1。L1 组 SOPs 比 B0 更高，说明结构参数缩减与全网络活动量变化并不等价。",'', '## 稳定性与配对统计','','| 模块 | Spearman(FR_A,D_B) | Spearman(L1_A,D_B) | Spearman(D_A,D_B) | 最低 25% D 重合率 |','|---|---:|---:|---:|---:|']
for r in stat:lines.append('| {} | {} | {} | {} | {} |'.format(r['module'],r['spearman_FR_A_vs_D_B'],r['spearman_L1_A_vs_D_B'],r['spearman_D_A_vs_D_B'],r['bottom_D_overlap']))
lines+=['', '完整各组逐样本差异及未经多重比较校正的配对 Bootstrap 95% CI 见 [metrics.json](metrics.json)；完整通道值和最低四分位 D_B 分布见 CSV。', '', '## 结构审计与资源', '', '真实模型目标 ATLIF 输出由对应 residual_function.3 消费；残差捷径未改变。独立构建的四组结构相对掩码在预先指定 32 张 A 图像上归一化 logits 误差均为 0、预测分歧为 0。此结果限于检查样本和当前浮点后端，并不保证所有输入逐位等价。所有原参数、BN 缓冲区的保留切片已逐项复核。实际 Conv 张量形状见 tensor_shapes.csv。', '', f"GPU 墙钟 {status['gpu_wall_seconds']:.2f} 秒；进程及已结束数据加载子进程累计 CPU {status['cpu_seconds']:.2f} 秒，GPU 峰值分配 {status['peak_gpu_allocated_bytes']} 字节。后处理 CPU 独立计时见资源记录。", '', '检查点及 SHA256 见 checkpoint_manifest.json；这些权重留在服务器，不提交 Git。没有训练、BN 重估、官方测试集或阈值手工更新。','', '## 局限与理论反馈','','单模型、单 checkpoint、固定剪枝率和单个随机种子；校准数据曾参与训练及前期校准；仅调查三个预设模块。单通道 D 不保证同时删除多个通道时的交互可加；结果不能证明一般结构剪枝无效。Bootstrap CI 未校正多重比较。SOP 为理论计数，并非硬件能耗或延迟。建议理论研究员审查 D 的样本稳定性和多通道交互解释，不自动增加剪枝率搜索或微调。']
lines += ['', '## 配对 Bootstrap（2000 次，seed=2027）', '', '| 比较 | 准确率差 | 95% CI | CE 差 | 95% CI |', '|---|---:|---|---:|---|']
for name,v in metrics['comparisons'].items():lines.append('| {} | {:.6f} | [{:.6f}, {:.6f}] | {:.6f} | [{:.6f}, {:.6f}] |'.format(name,v['correct']['delta'],*v['correct']['ci95'],v['loss']['delta'],*v['loss']['ci95']))
lines += ['', 'CI 为样本级配对区间，未作多重比较校正。P-Loss 相对 P-FR 的 SOP 节省比例低约 4.40 个百分点，也未满足同预算约束。']
(o/'REPORT.md').write_text('\n'.join(lines)+'\n');(o/'limitations.md').write_text('\n'.join(lines[lines.index('## 局限与理论反馈'):])+'\n')
(o/'postprocess_resources.json').write_text(json.dumps({'cpu_seconds_self':resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,'wall_seconds':time.monotonic()-start,'gpu_used':False},indent=2)+'\n')
print(json.dumps({'rows':rows,'statistics':stat,'gates':[metrics['H014_screening_gate'],metrics['H015_screening_gate']],'resources':json.loads((o/'postprocess_resources.json').read_text())},indent=2))
