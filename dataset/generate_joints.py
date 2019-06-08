import numpy as np
import cv2
import csv
import scipy.misc
import pickle

csv_file1 =  '/Users/aniki/Desktop/2019CVPR/LIP_project/LIP/txt/lip_train_set.csv'
csv_file2 =  '/Users/aniki/Desktop/2019CVPR/LIP_project/LIP/txt/lip_val_set.csv'

def generate_joint(split,file,joints):
    with open(file, "r") as input_file:
        '''
        generate dic of joint 16,3
        joint_weights:-1 no exist,0 exist and invisible,1 exist and visible
        '''
        for row in csv.reader(input_file):
#            print(row)
            im_name = row.pop(0)[:-4]
            joints[im_name] = np.zeros((16,3))
            image_path = '/Users/aniki/Desktop/2019CVPR/LIP_project/LIP/{}_images/{}.jpg'.format(split,im_name)
            img = scipy.misc.imread(image_path).astype(np.float)
            h = img.shape[0]#h,y,rows
            w = img.shape[1]#w,x,cols
#            print(row)
    
    
            for idx, point in enumerate(row):
                joint_id = (int)(idx/3)
#                print(point)
    
                if 'nan' in point:
                    joints[im_name][joint_id,2] = 0
    
                else:
                    if idx % 3 == 0:
                        w_ = int(point)  #x kuan
                        w_ = min(w_, w-1)
                        w_ = max(w_, 0)
                        joints[im_name][joint_id,0] = w_
                    elif idx % 3 == 1 :
                        h_ = int(point)#y gao
                        h_ = min(h_, h-1)
                        h_ = max(h_, 0)
                        joints[im_name][joint_id,1] = h_
                    elif idx % 3 == 2 :
                        if int(point)>0:
                            joints[im_name][joint_id,2] = -1
                        else:
                            joints[im_name][joint_id,2] = 1
    return joints
joints = {}
joints = generate_joint('train',csv_file1,joints)
joints = generate_joint('val',csv_file2,joints)
fw = open('/Users/aniki/Desktop/2019CVPR/LIP_project/LIP/txt/joints.pkl','wb')
pickle.dump(joints, fw)  
fw.close()



                        
