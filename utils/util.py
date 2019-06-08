import logging
import time
from pathlib import Path
import os
import torch
from utils.miou import compute_mean_ioU_file
import numpy as np  
from PIL import Image
import numpy as np
import torchvision


def create_logger(cfg,phase = 'train'):
    log_dir = Path(cfg.DIR.LOG)
    out_dir = Path(cfg.DIR.OUTPUT)
    vis_dir = Path(cfg.DIR.VIS)
    mk_dir(log_dir)
    mk_dir(out_dir)
    mk_dir(vis_dir)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = log_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    
    return logger,str(out_dir),str(vis_dir)



def acc(mask_pred,mask_gt):
    batch_size = mask_pred.shape[0]
    #mask_pred:batch_size,20,H,W
    #mask_gt:batch_size,H,W
    x = np.asarray(mask_pred,dtype = np.int32)
    y = np.asarray(mask_gt,dtype = np.int32)
    x = np.argmax(x,axis=1)
#    print(x)
    x = np.split(x,batch_size,0)
    y = np.split(y,batch_size,0)
#    print(len(x),len(y))
    x = np.squeeze(x)
    y = np.squeeze(y)
    acc = compute_mean_ioU_file(x,y)
    return acc


def mk_dir(path):
    if not path.exists():
        print('=> creating {}'.format(path))
        path.mkdir()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
def save_checkpoint(states,output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
#    if is_best and 'state_dict' in states:
#        torch.save(states['best_state_dict'],
#                   os.path.join(output_dir, 'model_best.pth'))
    


# colour map
COLORS = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor


def decode_parsing(labels, num_images=1, num_classes=21, is_pred=False):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    pred_labels = labels[:num_images].clone().cpu().data
    if is_pred:
        pred_labels = torch.argmax(pred_labels, dim=1)
    n, h, w = pred_labels.size()

    labels_color = torch.zeros([n, 3, h, w], dtype=torch.uint8)
    for i, c in enumerate(COLORS):
        c0 = labels_color[:, 0, :, :]
        c1 = labels_color[:, 1, :, :]
        c2 = labels_color[:, 2, :, :]

        c0[pred_labels == i] = c[0]
        c1[pred_labels == i] = c[1]
        c2[pred_labels == i] = c[2]

    return labels_color

def inv_preprocess(imgs, num_images):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    rev_imgs = imgs[:num_images].clone().cpu().data
    rev_normalize = NormalizeInverse(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    for i in range(num_images):
        rev_imgs[i] = rev_normalize(rev_imgs[i])

    return rev_imgs

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    

