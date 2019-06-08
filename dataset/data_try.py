from LIP_dataset import LIPDataSet
from torch.utils import data
import matplotlib.pyplot as plt
import torch

data_dir = '/Users/aniki/Desktop/2019CVPR/LIP_project/LIP'
split = 'trainval'
crop_size = 192,256
transform=None
if split is 'train':
    batch_size=8
else:
    batch_size=1

train_loader = data.DataLoader(LIPDataSet(data_dir, split, crop_size=crop_size,heatmap_size=[96,128], transform=transform),
                                  batch_size=batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True)

dataiter = iter(train_loader)
if split != 'test':
    img,label,heatmap,heatmap_w,meta = next(dataiter)
    im = img[0][:,:,(2,1,0)]
    label = label[0]
    heatmap_all = torch.zeros(128,96)
    for i in range(16):
    
        if i == 1:
            heatmap[0][i]=heatmap[0][i]
            heatmap_all = heatmap_all+heatmap[0][i]
        else:
            heatmap_all = heatmap_all+heatmap[0][i]
    
    #show
    plt.figure(figsize=(6,6),dpi=60) 
    
    plt.subplot(221)
    plt.imshow(im)
    plt.subplot(222)
    plt.imshow(label)
    plt.subplot(223)
    plt.imshow(heatmap_all)
    print('name:{},\n h:{},w:{},\n center:{},scale:{},\nrotation:{}'.format(meta['name'][0],meta['height'][0],meta['width'][0],meta['center'][0]
    ,meta['scale'][0],meta['rotation'][0]))
    print(im.shape)


else:
    img,meta = next(dataiter)
    im = img[0][:,:,(2,1,0)]
    plt.imshow(im)
    print('name:{},\n h:{},w:{},\n center:{},scale:{},\nrotation:{}'.format(meta['name'][0],meta['height'][0],meta['width'][0],meta['center'][0]
    ,meta['scale'][0],meta['rotation'][0]))
    print(im.shape)
    

    
    

