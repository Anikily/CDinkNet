import os
import numpy as np
import random
import torch
import cv2
from torch.utils import data
from dataset.transform import get_affine_transform
from dataset.transform import affine_transform
from dataset.transform import fliplr_joints
import pickle as pkl

'''
the new datase for LIP challenge
new property:
    1.rotation of data
    2.test dataset
    3.change the method of transform and load image
    4.change the generation of heatmap
    5.enlarge the pic range in output
    
    reference:https:1.//github.com/Microsoft/human-pose-estimation.pytorch
              https://github.com/liutinglt/CE2P/blob/master/dataset/datasets.py

'''


class LIPDataSet(data.Dataset):
    def __init__(self, root, dataset, crop_size=[384,512], heatmap_size=[96,128], scale_factor=0.25,
                 rotation_factor=30,ignore_label=255, transform=None):
        """
        :rtype:
        """
        #data_dir, data id, joints
        self.root = root
        self.dataset = dataset
        list_path = os.path.join(self.root, 'txt',self.dataset + '_id.txt')
        self.im_list = [i_id.strip() for i_id in open(list_path)]
        self.joints = self.get_joints()
        #joints:dictionary,key:im_name,16,3,(w,h)
        
        self.number_samples = len(self.im_list)
        
        self.aspect_ratio = crop_size[0] * 1.0 / crop_size[1]
        
        self.target_type = 'gaussian'
        self.heatmap_size = heatmap_size
        self.num_joints = 16
        self.sigma=2
        #transform of data: 1.scale 2.rotation 3.flip 4.resize 5.transform
        self.crop_size = np.asarray(crop_size)
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = 0.5
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [11, 14], [12, 13], [10, 15]]
        self.transform = transform
        self.ignore_label = ignore_label

    def __len__(self):
        return self.number_samples
    
    def get_joints(self):
        joint_path = os.path.join(self.root, 'txt','joints.pkl')
        fr = open(joint_path,'rb')
        data = pkl.load(fr)
        return data

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        '''rescale the shorter to the aspect-ratio
           and get the center
        '''
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale
    def generate_heatmap(self, joints, joints_vis):
            '''
            :param joints:  [num_joints, 2]
            :param joints_vis: [num_joints],-1:none,0:invisible,1:visible
            :return: target,target_weights
            '''
            target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
            target_weight = joints_vis
    
            assert self.target_type == 'gaussian', \
                'Only support gaussian map now!'
    
            if self.target_type == 'gaussian':
                target = np.zeros((self.num_joints,
                                   self.heatmap_size[1],
                                   self.heatmap_size[0]),
                                  dtype=np.float32)
    
                tmp_size = self.sigma * 3
    
                for joint_id in range(self.num_joints):
                    feat_stride = self.crop_size / self.heatmap_size
                    mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                    mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                    # Check that any part of the gaussian is in-bounds
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                            or br[0] < 0 or br[1] < 0:
                        # If not, just return the image as is
                        target_weight[joint_id] = 0
                        continue
    
                    # # Generate gaussian
                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2
                    # The gaussian is not normalized, we want the center value to equal 1
                    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))
    
                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])
    
                    v = target_weight[joint_id]
                    if v > 0:
                        target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    
            return target, target_weight


    def __getitem__(self, index):
        # Load training image
        im_name = self.im_list[index]
        im_path = os.path.join(self.root, self.dataset + '_images', im_name + '.jpg')
        parsing_anno_path = os.path.join(self.root, self.dataset + '_segmentations', im_name + '.png')
        in_ = cv2.imread(im_path, cv2.IMREAD_COLOR)
        in_ = in_[:,:,::-1]
        h_, w_, _ = in_.shape
        im = cv2.resize(in_,(self.crop_size[0],self.crop_size[1]),interpolation=cv2.INTER_LINEAR)
#        trans_resize = cv2.getAffineTransform(np.array([[0,0],[h_//2,w_//2],[h_,w_]]),
#            np.array([[0,0],[self.crop_size[1]//2,self.crop_size[0]//2],[self.crop_size[1],self.crop_size[0]]]))
        trans_resize = cv2.getAffineTransform(np.float32([[0,0],[w_,h_],[0,h_]]), np.float32([[0,0],[self.crop_size[0],self.crop_size[1]],[0,self.crop_size[1]]]))
        #pre mask and heatmap
        h, w, _ = im.shape
        # parsing_anno = np.zeros((h_, w_), dtype=np.long)
        # heatmap = np.zeros((16,self.heatmap_size[1],self.heatmap_size[0]))

        # Get center and scale,:w,h
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if self.dataset != 'test': 
            #define transformation when trainining 
            parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)
            parsing_anno = cv2.resize(parsing_anno,(self.crop_size[0],self.crop_size[1]),interpolation=cv2.INTER_NEAREST)
            # joints = self.joints[im_name][:,0:2]
            # joints_weights = self.joints[im_name][:,2]
            # for i in range(self.num_joints):
                # if joints_weights[i] >= 0.0:
                    
                    # joints[i, :] = affine_transform(joints[i, :], trans_resize)

            if self.dataset == 'train' or self.dataset == 'trainval':

                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                #scale:rescale data in range(1+sf,1-sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                    if random.random() <= 0.6 else 0
                #rotation:ratate data in range(r*2,-r*2)

                if random.random() <= self.flip_prob:
#                    print('flip')
                    im = im[:, ::-1, :]
                    center[0] = im.shape[1] - center[0] - 1
                    #flip mask
                    parsing_anno = parsing_anno[:, ::-1]
                    right_idx = [15, 17, 19]
                    left_idx = [14, 16, 18]
                    for i in range(0, 3):
                        right_pos = np.where(parsing_anno == right_idx[i])
                        left_pos = np.where(parsing_anno == left_idx[i])
                        parsing_anno[right_pos[0], right_pos[1]] = left_idx[i]
                        parsing_anno[left_pos[0], left_pos[1]] = right_idx[i]
                    #flip joints
                    # joints, joints_weights = fliplr_joints(
                    # joints, joints_weights, w, self.flip_pairs)

        trans = get_affine_transform(center, s, r, self.crop_size)

#        trans1 = get_affine_transform(center, s, r, self.heatmap_size)
        meta = {
            'name': im_name,
            'center': center,
            'height': h_,
            'width': w_,
            'scale': s,
            'rotation': r
        }
        

            
        if self.dataset == 'test':
            if self.transform:
                img = self.transform(im)
            return img, meta

        else:
            input = cv2.warpAffine(
                im,
                trans,
                (int(self.crop_size[0]), int(self.crop_size[1])),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0))

            if self.transform:
                input = self.transform(input)

            #transform mask and joints for heatmap
            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[0]), int(self.crop_size[1])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255))
#            print('before transoform head point:{}'.format(joints[8]))
#            print(trans,trans_resize)
            # for i in range(self.num_joints):
                # if joints_weights[i] >= 0.0:
                    
                    # joints[i, :] = affine_transform(joints[i, :], trans)
            label_parsing = torch.from_numpy(label_parsing)
#            print('after transoform head point:{}'.format(joints[8]))
            # heatmap,heatmap_w = self.generate_heatmap(joints,joints_weights)
            # heatmap = torch.from_numpy(heatmap)


            return input, label_parsing, 0, 1, meta