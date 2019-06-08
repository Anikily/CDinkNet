import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F



class CriterionAll(nn.Module):
    def __init__(self, ignore_index=255):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
   
    def parsing_loss(self, preds, target):
        h, w = target.size(1), target.size(2)

        loss = 0

        # loss for parsing
        #print(len(preds))
        if len(preds) != 5:
            preds_parsing = preds

            scale_pred = F.interpolate(input=preds_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += self.criterion(scale_pred, target)
            return loss

        else:
            for i,pred in enumerate(preds):
                if i == 0:
                    weight = 1
                else:
                    weight = 0.5

                preds_parsing = pred
                scale_pred = F.interpolate(input=preds_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
                loss += self.criterion(scale_pred, target)*weight
            return loss



    def forward(self, preds, target):
          
        loss = self.parsing_loss(preds, target) 
        return loss
    
