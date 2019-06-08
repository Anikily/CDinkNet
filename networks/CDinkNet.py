import torch

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from functools import partial
from networks.backbone import *
from config import config
affine_par = True
import functools
import sys, os

from libs import InPlaceABN, InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

nonlinearity = partial(F.relu,inplace=False)

def conv(inplanes,outplanes,kernel=3,stride=1,padding=1):
    conv = nn.Sequential(nn.Conv2d(inplanes,outplanes,kernel_size=kernel,stride=stride,padding=padding),
                         InPlaceABNSync(outplanes))
    return conv
def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0):
    deconv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
                         InPlaceABNSync(out_channels))
    return deconv
class Smooth_layer(nn.Module):
    def __init__(self,filters):
        super(Smooth_layer, self).__init__()
        self.filters = filters
        self.smooth51 = DecoderBlock(self.filters[4],self.filters[3])
        self.smooth52 = DecoderBlock(self.filters[3],self.filters[1])

        
        self.smooth41 = DecoderBlock(self.filters[3],self.filters[2])
        self.smooth42 = DecoderBlock(self.filters[2],self.filters[1])
        
        self.smooth31 = DecoderBlock(self.filters[2],self.filters[1])

        self.smooth2 = conv(filters[1],filters[1],3,1,1)

        self.smooth_all = nn.Sequential(conv(filters[3],filters[2],1,1,0),
                                        conv(filters[2],filters[1],3,1,1),
                                        conv(filters[1],filters[1]*2,1,1,0))
    def forward(self,p5,p4,p3,p2):
        k5 = self.smooth51(p5)
        k5 = self.smooth52(k5)

        k4 = self.smooth41(p4)
        k4 = self.smooth42(k4)

        k3 = self.smooth31(p3)

        k2 = self.smooth2(p2)

        out = torch.cat((k5,k4,k3,k2),1)
        out = self.smooth_all(out)
        return out
class ASPPModule(nn.Module):
    """
    Reference: 
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    x:2048-->1024-->256-->1024-->2048
    """
    
    def __init__(self, features, inner_features=256, out_features=512, dilations=(12, 24, 36)):
        super(ASPPModule, self).__init__()
        self.ex_conv = conv(features,features//2,1,1,0)

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                   nn.Conv2d(features//2, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv2 = nn.Sequential(nn.Conv2d(features//2, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features//2, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features//2, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features//2, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   InPlaceABNSync(inner_features))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(features),
            nn.Dropout2d(0.1)
            )
        
    def forward(self, x):

        _, _, h, w = x.size()
        x = self.ex_conv(x)

        feat1 = F.upsample(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = InPlaceABNSync(in_channels // 4)


        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = InPlaceABNSync(in_channels // 4)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = InPlaceABNSync(n_filters)


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        return x


class CDinkNet_ASPP(nn.Module):
    def __init__(self, cfg):
        super(CDinkNet_ASPP, self).__init__()
        self.backbone = eval('resnet{}'.format(config.MODEL.PRETRAIN_LAYER))()
        self.filters = [64, 256, 512, 1024, 2048]
        #dink_module
        self.aspp = ASPPModule(self.filters[4])
        self.out5 = nn.Sequential(conv(self.filters[4],self.filters[3]),conv(self.filters[3],self.filters[2]),
            conv(self.filters[2],self.filters[1]),nn.Conv2d(self.filters[1],20,1,1,0))
        #decoder
        self.decoder4 = nn.Sequential(conv(self.filters[4],self.filters[4]//4,1,1,0),
                                      conv(self.filters[4]//4,self.filters[4]//4,3,1,1),
                                      conv(self.filters[4]//4,self.filters[3],1,1,0))
        self.out4 = nn.Sequential(conv(self.filters[3],self.filters[2]),
            conv(self.filters[2],self.filters[1]),nn.Conv2d(self.filters[1],20,1,1,0))

        self.decoder3 = DecoderBlock(self.filters[3], self.filters[2])
        self.out3 = nn.Sequential(conv(self.filters[2],self.filters[1]),nn.Conv2d(self.filters[1],20,1,1,0))

        self.decoder2 = DecoderBlock(self.filters[2], self.filters[1])
        self.out2 = nn.Sequential(conv(self.filters[1],self.filters[0]),nn.Conv2d(self.filters[0],20,1,1,0))

        #h/4,w/4,256
        self.smooth_layer = Smooth_layer(self.filters)

        self.finaldeconv1 = deconv(self.filters[2], self.filters[1], 4, 2, 1)
        self.finaldeconv2 = deconv(self.filters[1], 128, 4, 2, 1)


        self.finalconv2 = conv(128,64,3,1,1)
        self.finalconv3 = nn.Conv2d(64,cfg.NUM_CLASS,1,1,0)
        
    def forward(self,x):
        #encoder
        c2,c3,c4,c5 = self.backbone(x)
        #center
        p5 = self.aspp(c5)
        out5 = self.out5(p5)
        #decoder
        p4 = self.decoder4(p5)+c4#16/1
        out4 = self.out4(p4)

        p3 = self.decoder3(p4)+c3#8/1
        out3 = self.out3(p3)
        p2 = self.decoder2(p3)+c2#4/1
        out2 = self.out2(p2)
        #smooth layer

        out = self.smooth_layer(p5,p4,p3,p2) #h/4,2/4,512

        #final_layer
        out = self.finaldeconv1(out)
        out = self.finaldeconv2(out)
        out = self.finalconv2(out)
        out = self.finalconv3(out)
        return [out,out2,out3,out4,out5]


        
        
        
        
        
        
        
        
        
        
        
        
        
        
