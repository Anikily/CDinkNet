import torch
import torch.nn as nn
from config import config
from utils.TTA_back import *
import os
import numpy as np
import cv2

def get_avg(outs):
    num = len(outs)
    out = np.zeros_like(outs[0])

    for i in outs:
        out+=i
    return out/num

def get_rev_out(imgs,model,output,h,w):

    # print('get inv img')
    img_inv = imgs['img_inv'][0].cuda()
    img_out_inv = model(img_inv)
    img_out_inv = nn.functional.interpolate(input=img_out_inv,size=(h,w),mode='bilinear',align_corners=True)
    img_out_inv = inv_back([img_out_inv])
    output.append(img_out_inv[0])
    return output

def get_rotate_out(imgs,model,output,h,w):

    # print('get rotation img')
    r_outs = []
    r_imgs = imgs['rotation']
    for img in r_imgs:
        r_out = model(img.cuda())
        r_out = nn.functional.interpolate(input=r_out,size=(384,384),mode='bilinear',align_corners=True)
        r_outs.append(r_out)

    r_outs = rotation_back(r_outs)#numpy
    r_output = get_avg(r_outs)
    r_output = np.transpose(r_output,(2,0,1))
    r_output = torch.from_numpy(r_output).unsqueeze(0).cuda()
    r_output = nn.functional.interpolate(input=r_output,size=(h,w),mode='bilinear',align_corners=True)
    output.append(r_output)

    # print('get rotation_inv img')
    r_outs_inv = []
    r_imgs_inv = imgs['rotation_inv']
    for img in r_imgs_inv:
        
        r_out_inv = model(img.cuda())
        r_out_inv = nn.functional.interpolate(input=r_out_inv,size=(384,384),mode='bilinear',align_corners=True)
        r_outs_inv.append(r_out_inv)
        
    r_outs_inv = inv_back(r_outs_inv)
    r_outs_inv = rotation_back(r_outs_inv)#numpy
    r_output_inv = get_avg(r_outs_inv)
    r_output_inv = np.transpose(r_output_inv,(2,0,1))
    r_output_inv = torch.from_numpy(r_output_inv).unsqueeze(0).cuda()
    r_output_inv = nn.functional.interpolate(input=r_output_inv,size=(h,w),mode='bilinear',align_corners=True)
    output.append(r_output_inv)
    return output



def get_shift_out(imgs,model,output,h,w):
    if config.TEST.SHIFT==True:
        s_outs = []
        s_imgs = imgs['shift']
        # print('get shift img')
        for i,img in enumerate(s_imgs):

            s_out = model(img.cuda())
            s_out = nn.functional.interpolate(input=s_out,size=(384,384),mode='bilinear',align_corners=True)   
            s_outs.append(s_out)

        s_outs = shift_back(s_outs)
        s_output = get_avg(s_outs)
        s_output = np.transpose(s_output,(2,0,1))
        s_output = torch.from_numpy(s_output).unsqueeze(0).cuda()
        s_output = nn.functional.interpolate(input=s_output,size=(h,w),mode='bilinear',align_corners=True)
        output.append(s_output)

        # print('get shift_inv img')
        s_outs_inv = []
        s_imgs_inv = imgs['shift_inv']
        for i,img in enumerate(s_imgs_inv):

            s_out_inv = model(img.cuda())
            s_out_inv = nn.functional.interpolate(input=s_out_inv,size=(384,384),mode='bilinear',align_corners=True)
            s_outs_inv.append(s_out_inv)
        s_outs_inv = inv_back(s_outs_inv)
        s_outs_inv = shift_back(s_outs_inv)
        s_output_inv = get_avg(s_outs_inv)
        s_output_inv = np.transpose(s_output_inv,(2,0,1))
        s_output_inv = torch.from_numpy(s_output_inv).unsqueeze(0).cuda()
        s_output_inv = nn.functional.interpolate(input=s_output_inv,size=(h,w),mode='bilinear',align_corners=True)
        output.append(s_output_inv)
    return output