# coding:utf-8
import numpy as np
from easydict import EasyDict as edict
import torch

config = edict()

# config GENERAL
config.NUM_POSE = 16
config.NUM_CLASS = 20
config.GPUS = '0,1'
config.NUM_WORKERS = 16


# config DIR
config.DIR = edict()
config.DIR.MAIN = ''
config.DIR.DATA = '{}/LIP'.format(config.DIR.MAIN)
config.DIR.OUTPUT = './checkpoint'.format(config.DIR.MAIN)
config.DIR.LOG = './log'
config.DIR.VIS = './vis/'
config.DIR.TEST_MODEL = ''


# config DATA
config.DATA = edict()
config.DATA.SPLIT = 'train'
config.DATA.TYPE = torch.float32
config.DATA.INPUT_SIZE = '192,256'
config.DATA.BATCH_SIZE = 4

# model
config.MODEL = edict()
config.MODEL.NAME = 'CDinkNet_ASPP'
config.MODEL.PRETRAIN_LAYER = 101
if config.MODEL.PRETRAIN_LAYER == 50 | config.MODEL.PRETRAIN_LAYER == 101 | config.MODEL.PRETRAIN_LAYER == 152:
    config.MODEL.FILTERS = [64, 256, 512, 1024, 2048]
elif config.MODEL.PRETRAIN_LAYER == 34:
    config.MODEL.FILTERS = [64, 64, 128, 256, 512]


# visulize


# train
config.TRAIN = edict()
config.TRAIN.RESUME = False
config.TRAIN.PRE = ''
# train.opt
config.TRAIN.LR = 0.001
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0
# train.learning_schedual
config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 120
config.TRAIN.POLY_POWER = 0.7
config.TRAIN.WEIGHT_DECAY = 0.0005
