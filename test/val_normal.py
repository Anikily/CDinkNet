import torch
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms


from dataset.LIP_dataset_val import LIPDataSet
from networks.DinkNet_atrous import DinkNet_atrous
from networks.ASPP_DinkNet import ASPP_DinkNet
#from networks.CPN_parsing import CDinkNet_ASPP
from networks.old_CDinkNet_ASPP import CDinkNet_ASPP
from networks.ACDinkNet import ACDinkNet
from networks.DinkNet_ASPP import DinkNet50

from networks.backbone import *
from config import config
from utils.utils import create_logger,AverageMeter,save_checkpoint,acc
from utils.utils import decode_parsing, inv_preprocess
from utils.TTA_back import *
import os
import numpy as np
import cv2
import time
config.MODEL.PRETRAIN_LAYER = 101
test_model = '/home/aniki/done/CDinkNet_101/CDinkNet_101/checkpoint/checkpoint_parallel_deconv.pth'
#test_model = '/home/aniki/done/DinkNet_ASPP_101/checkpoint/checkpoint_parallel_deconv.pth'
#test_model = '/home/aniki/done/CPN101/checkpoint6/checkpoint_parallel_deconv.pth'
data_dir = '/home/aniki/LIP_project/LIP'



w, h = 192,256


input_size = [w, h]
heatmap_size = [w//4,h//4]
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

test_loader = data.DataLoader(LIPDataSet(config,data_dir, 'val', crop_size=input_size, heatmap_size=heatmap_size, transform=transform),
                              batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS,
                              pin_memory=True)



def get_result(x,h,w):
    x = nn.functional.interpolate(input=x,size=(h,w),mode='bilinear',align_corners=True)
    x = x.squeeze(0).cpu().numpy()
    print(x.shape)
    print(h,w)
	# out = np.asarray(out,dtype=np.float32)
    result = np.asarray(np.argmax(x, axis=0), dtype=np.uint8)
	# result = np.asarray(result,dtype = np.int64)

    return result


def val(cfg,model):
    for i,(imgs,parsing,meta) in enumerate(test_loader):
        img = imgs['img'][0].cuda()
        img_inv = np.flip(img.cpu().numpy(),3).copy()
        img_inv = torch.from_numpy(img_inv).cuda()
        h = meta['height'][0]
        w = meta['width'][0]
        id_ = meta['name'][0]
        print('{} pricure is processing'.format(id_))
        out = model(img)
        out = nn.functional.interpolate(input=out,size=(h,w),mode='bilinear',align_corners=True)
        if cfg.TEST.INV ==False:
            print('get inv')
            img_out_inv = model(img_inv)
            img_out_inv = nn.functional.interpolate(input=img_out_inv,size=(h,w),mode='bilinear',align_corners=True)
            img_out_inv = inv_back([img_out_inv])
            out = (out+img_out_inv[0])*0.5
        result = get_result(out,h,w)
        print('result:{}'.format(result.shape))
   
        img_name = 'val'+'/'+str(id_)+'.png'
        print('saving in {}'.format(img_name))
        print('{}/{}'.format(i,len(test_loader)))
        print()
        cv2.imwrite(img_name,result)



cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

model_state_file = test_model
checkpoint = torch.load(model_state_file)
print('==>loading {} from {}'.format(checkpoint['model'],model_state_file))

model = eval(checkpoint['model'])(config)
print('build {} model successiful'.format(checkpoint['model']))
print(checkpoint['epoch'])
model.load_state_dict(checkpoint['module_state_dict'])
model = model.cuda()
#model = DataParallelModel(deeplab)
#model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
print('initalize model succesiful.')
model.eval()
with torch.no_grad():
	val(config,model)
	print('finish')
