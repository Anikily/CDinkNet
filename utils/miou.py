
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 17:19:26 2019

@author: yan
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:40:40 2019

@author: yan
"""

import numpy as np
import cv2
import os
import json
from collections import OrderedDict
from PIL import Image as PILImage

#class name for 20 classes 
LABELS = ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat', \
          'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm', 'Left-leg',
          'Right-leg', 'Left-shoe', 'Right-shoe']

def get_confusion_matrix(gt_label, pred_label, num_classes):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param num_classes: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * num_classes + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i_label in range(num_classes):
        for i_pred_label in range(num_classes):
            cur_index = i_label * num_classes + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix

def compute_mean_ioU_file(gt_list,pred_list,num_classes=20):
    
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(0,len(gt_list)):
        gt = gt_list[i]
        pred = pred_list[i]

        ignore_index = gt != 255
        
        gt = gt[ignore_index]
        pred = pred[ignore_index]
    
        confusion_matrix += get_confusion_matrix(gt, pred, num_classes)

    pos = confusion_matrix.sum(1)#hang-> pre
    res = confusion_matrix.sum(0)# lie -> gt true
    tp = np.diag(confusion_matrix) #zhen zheng 

    pixel_accuracy = (tp.sum() / pos.sum())*100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean())*100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp)) * 100
    mean_IoU = IoU_array.mean()
    fw_IoU = ((pos*IoU_array)*(1/pos.sum())).sum()
    
#    print('Pixel accuracy: %f \n' % pixel_accuracy)
#    print('Mean accuracy: %f \n' % mean_accuracy)
#    print('Mean IU: %f \n' % mean_IoU)
#    print('Frequency weighted IoU: %f \n' % fw_IoU)
    return mean_IoU

#if __name__ == "__main__":
 #   palette = get_palette(20)
#    num_classes = 20
#    out = compute_mean_ioU_file(pre_list, num_classes, gt_list)
#    print("now iou result is ", out)
