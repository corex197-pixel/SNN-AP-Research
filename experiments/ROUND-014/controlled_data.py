"""CIFAR-10 train-file-only loader; never constructs torchvision CIFAR10 or accesses test_batch."""
import pickle,json,hashlib,random
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.augment import CIFAR10Policy,Cutout
ROOT=Path(__file__).resolve().parents[2]; DATA=Path('/home/hym/dataset/cifar10/cifar-10-batches-py')
def transform(training):
 norm=transforms.Normalize((.4914,.4822,.4465),(.2023,.1994,.2010))
 return transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),CIFAR10Policy(),transforms.ToTensor(),norm,Cutout(n_holes=1,length=16)]) if training else transforms.Compose([transforms.ToTensor(),norm])
class TrainFileOnly(Dataset):
 def __init__(self,indices,training=True):
  self.ids=list(indices);self.transform=transform(training);self.images=[];self.labels=[]
  for i in range(1,6):
   with (DATA/f'data_batch_{i}').open('rb') as f:d=pickle.load(f,encoding='bytes')
   self.images.append(d[b'data'].reshape(-1,3,32,32));self.labels.extend(d[b'labels'])
  self.images=np.concatenate(self.images);assert len(self.labels)==len(self.images)==50000 and all(0<=i<50000 for i in self.ids)
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):
  index=self.ids[i];im=Image.fromarray(self.images[index].transpose(1,2,0));return self.transform(im),self.labels[index],index
