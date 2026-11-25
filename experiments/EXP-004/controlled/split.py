import sys,json,hashlib,collections
from pathlib import Path
import numpy as np
from torchvision.datasets import CIFAR10
from core import DATA,SPLIT
if SPLIT.exists():sys.exit('split exists; refusing overwrite')
ds=CIFAR10(DATA,train=True,download=False);assert ds._check_integrity() and len(ds)==50000
labels=np.asarray(ds.targets);rng=np.random.RandomState(2026);parts={'train':[],'validation':[],'diagnostic':[]}
for c in range(10):
 x=np.flatnonzero(labels==c);assert len(x)==5000;rng.shuffle(x)
 for key,selection in [('train',x[:4500]),('validation',x[4500:4950]),('diagnostic',x[4950:])]:parts[key].extend(map(int,selection))
for key in parts:parts[key].sort()
g=np.random.RandomState(2026);diag256=sorted(map(int,g.choice(parts['diagnostic'],size=256,replace=False)))
sets=[set(parts[k]) for k in parts];assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]);assert set.union(*sets)==set(range(50000))
sha=lambda b:hashlib.sha256(b).hexdigest()
source=Path(DATA)/'cifar-10-batches-py';checks={f.name:sha(f.read_bytes()) for f in sorted(source.glob('data_batch_*'))}
result={**parts,'diagnostic_256':diag256,'metadata':{'split_seed':2026,'diagnostic_seed':2026,'algorithm':'numpy RandomState MT19937; class-wise permutation in ascending class order; first 4500 train, next 450 validation, last 50 pool; sorted indices; separate RandomState choice 256 without replacement from sorted pool','counts':{k:len(v) for k,v in parts.items()},'per_class':{k:dict(collections.Counter(map(int,labels[v]))) for k,v in parts.items()},'train_batch_sha256':checks,'torchvision_integrity':True,'data_source':'CIFAR-10 official training partition','test_used_for_split':False}}
SPLIT.parent.mkdir(exist_ok=True,parents=True);SPLIT.write_text(json.dumps(result,separators=(',',':'))+'\n');print('sha256',sha(SPLIT.read_bytes()),result['metadata'])
