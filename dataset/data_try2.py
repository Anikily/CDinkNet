import sys
sys.path.append('/Users/aniki/Desktop/2019CVPR/LIP_project/code/parsing_code')
from LIP_dataset import LIPDataSet
from torch.utils import data
import matplotlib.pyplot as plt
import torch
import time
from tools.utils import AverageMeter

data_dir = '/Users/aniki/Desktop/2019CVPR/LIP_project/LIP'
split = 'train'
crop_size = 512,384
transform=None
if split is 'train':
    batch_size=8
else:
    batch_size=1

train_loader = data.DataLoader(LIPDataSet(data_dir, split, crop_size=crop_size,heatmap_size=[96,128], transform=transform),
                                  batch_size=batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True)

batch_time = AverageMeter()
end = time.time()
for i,(x,y,z,a,b) in enumerate(train_loader):
    batch_time.update(time.time()-end)
    print('data_load time for one batch i {}'.format(batch_time.val))
    print('{}/{}'.format(i,len(train_loader)))
    print(x.shape)
    print(y.shape)
    print(z.shape)
    print(z.max())
    end = time.time()

    
print('avg of batch_time is {}'.format(batch_time.avg))
print('totle time is {}'.format(batch_time.sum))

