import torch
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
import sys
sys.path.append('/home/aniki/code')



from dataset.LIP_dataset import LIPDataSet
from dataset.transform import *
from networks.DinkNet_atrous_early import DinkNet_atrous
from networks.ASPP_DinkNet import ASPP_DinkNet
from networks.CDinkNet import CDinkNet
from networks.CDinkNet_ASPP_new import CDinkNet_ASPP
from networks.ACDinkNet import ACDinkNet

from networks.backbone import *
from config import config
from util import decode_parsing
from TTA_back import *
import os
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

test_model = '/home/aniki/done/submit/CDinkNet_ASPP/checkpoint1/checkpoint_parallel111.pth'
data_dir = '/home/aniki/LIP_project/LIP'



w, h = 384,384
input_size = [w, h]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])





def get_result(x,h,w):
	x = x.squeeze(0).cpu().numpy()
	#print(x.shape)
	out = np.argmax(x,0)
	#print(out.dtype,h,w)
	# out = np.asarray(out,dtype=np.float32)
	result = cv2.resize(out,(w,h),interpolation = cv2.INTER_NEAREST)
	# result = np.asarray(result,dtype = np.int64)

	return result
def get_data_list():
    list_path = '/home/aniki/code/test/test_data_list.txt'
    im_list = [i_id.strip() for i_id in open(list_path)]
 
    return im_list

def get_avg(outs):
    num = len(outs)
    out = np.zeros_like(outs[0])
    #print(out.shape)
    for i in outs:
        out+=i
    return out/num

def test(image,model):
        #get image
    im = cv2.imread(image,cv2.IMREAD_COLOR)
    H,W,_ = im.shape
    im = cv2.resize(im,(w,h),interpolation=cv2.INTER_LINEAR)
    imgs = {}
    if config.TEST.TTA==True:
        imgs = TTA(config,im)
        if transform:
            for key in imgs.keys():
                for i,img in enumerate(imgs[key]):
                    imgs[key][i] = transform(imgs[key][i])
    else: 
        if transform:
            imgs['img'] = [transform(im)]
        else:
            imgs['img'] = im

    img = imgs['img'][0].unsqueeze(0)


    img = img.float().cuda()
    out = model(img)
    out = nn.functional.interpolate(input=out,size=(384,384),mode='bilinear',align_corners=True)

    if config.TEST.INV ==True:
        img_inv = imgs['img_inv'][0].unsqueeze(0).cuda()
        img_out_inv = model(img_inv)
        img_out_inv = nn.functional.interpolate(input=img_out_inv,size=(384,384),mode='bilinear',align_corners=True)
        img_out_inv = inv_back([img_out_inv])
        out = (out+img_out_inv[0])* 0.5

        if config.TEST.ROTATION==True:
            r_outs = []
            r_imgs = imgs['rotation']
            for i,img in enumerate(r_imgs):
                r_out = model(img.unsqueeze(0).cuda())
                r_out = nn.functional.interpolate(input=r_out,size=(384,384),mode='bilinear',align_corners=True)
                r_outs.append(r_out)
            r_outs = rotation_back(r_outs)#numpy
            r_output = get_avg(r_outs)

            r_output = np.transpose(r_output,(2,0,1))
            r_output = torch.from_numpy(r_output).unsqueeze(0).cuda()

            if config.TEST.INV ==True:
                r_outs_inv = []
                r_imgs_inv = imgs['rotation_inv']
                for i,img in enumerate(r_imgs_inv):



                    r_out_inv = model(img.unsqueeze(0).cuda())
                    r_out_inv = nn.functional.interpolate(input=r_out_inv,size=(384,384),mode='bilinear',align_corners=True)



                    r_outs_inv.append(r_out_inv)
                r_outs_inv = inv_back(r_outs_inv)
                r_outs_inv = rotation_back(r_outs_inv)#numpy

                r_output_inv = get_avg(r_outs_inv)
                r_output_inv = np.transpose(r_output_inv,(2,0,1))
                r_output_inv = torch.from_numpy(r_output_inv).unsqueeze(0).cuda()




    if config.TEST.SHIFT==True:
        s_outs = []
        s_imgs = imgs['shift']
        for i,img in enumerate(s_imgs):

            #print('aaa')
            #print('when shift,the shape of input is :{}'.format(img.shape))

            s_out = model(img.unsqueeze(0).cuda())
            s_out = nn.functional.interpolate(input=s_out,size=(384,384),mode='bilinear',align_corners=True)   
  
            #print(',the shape output is {}'.format(s_out.shape))
#            show_pic(image,i,'before,',s_out)
            #print(s_out.shape)       

            s_outs.append(s_out)
        s_outs = shift_back(s_outs)

        for i,img in enumerate(s_outs):
            img = torch.from_numpy(np.transpose(img,(2,0,1))).unsqueeze(0)
            #print('after shift back')
            #print(img.shape)
#            show_pic(image,i,'after',img)

        s_output = get_avg(s_outs)
        s_output = np.transpose(s_output,(2,0,1))
        s_output = torch.from_numpy(s_output).unsqueeze(0).cuda()

        if config.TEST.INV ==True:
            s_outs_inv = []
            s_imgs_inv = imgs['shift_inv']
            for i,img in enumerate(s_imgs_inv):

                #print('aaa')
                #print('when shift,the shape of input is :{}'.format(img.shape))

                s_out_inv = model(img.unsqueeze(0).cuda())
                s_out_inv = nn.functional.interpolate(input=s_out_inv,size=(h,w),mode='bilinear',align_corners=True)

                #print(',the shape output is {}'.format(s_out_inv.shape))
#                show_pic(image,i,'before,',s_out_inv)
                #print(s_out_inv.shape)      

                s_outs_inv.append(s_out_inv)
            s_outs_inv = inv_back(s_outs_inv)
            s_outs_inv = shift_back(s_outs_inv)

            for i,img in enumerate(s_outs_inv):
                img = torch.from_numpy(np.transpose(img,(2,0,1))).unsqueeze(0)
                #print('after shift back')
                #print(img.shape)
#                show_pic(image,i,'after',img)

            s_output_inv = get_avg(s_outs_inv)
            s_output_inv = np.transpose(s_output_inv,(2,0,1))
            s_output_inv = torch.from_numpy(s_output_inv).unsqueeze(0).cuda()

    out = (out+r_output+r_output_inv+s_output+s_output_inv)/5


    #print(out.shape)
    result = decode_parsing(out, num_images=1, num_classes=20, is_pred=True)
    #print(result.type(),result.shape)
    #print(result.shape)
    result = result.squeeze(0)
    result = np.asarray(result)
    result = np.transpose(result,(1,2,0))
    result = cv2.resize(result,(W,H),interpolation=cv2.INTER_NEAREST)
    # result = get_result(result,H,W)
    #print(result.shape)
    
    out_name = '/home/aniki/code/test/out'+str(image[1:-4])+'.png'
    print(out_name)
    cv2.imwrite(out_name,result)




def show_pic(image,i,phase,out):
    #print('keeping {} pic ing...'.format(i))
    result = decode_parsing(out, num_images=1, num_classes=20, is_pred=True)
    result = result.squeeze(0)
    result = np.asarray(result)
    result = np.transpose(result,(1,2,0))
    out_name = '/home/aniki/code/test/out/{}'.format(phase)+str(image[11:22])+str(i)+'.png'.format(phase)
    cv2.imwrite(out_name,result)




cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

model_state_file = test_model
checkpoint = torch.load(model_state_file)
print('==>loading {} from {}'.format(checkpoint['model'],model_state_file))

model = eval(checkpoint['model'])(config)
print('build {} model successiful'.format(checkpoint['model']))
model.load_state_dict(checkpoint['module_state_dict'])
model = model.cuda()
#model = DataParallelModel(deeplab)
#model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
print('initalize model succesiful.')
model.eval()
with torch.no_grad():
    im_list = get_data_list()
    print(im_list)
    for j,i in enumerate(im_list):
        print(f'{j}/{len(im_list)}')
        im_name = i
        print('{} pricure is processing'.format(im_name))
        test(im_name,model)



print('finish')
