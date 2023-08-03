import os.path as osp
import os
import cv2
import numpy as np
import imageio
import glob
from multiprocessing import Pool
from tqdm import tqdm
import random
img_suffix      = '_leftImg8bit.png'                #img
seg_map_suffix  = '_gtFine_labelTrainIds.png'       #label
data_root       = '/SSD_DISK/users/kuangshaochen/waymo_test/waymo_val_dataset'
destination     = '/HDD_DISK/datasets/data/cityscapes'

waymo_full = destination + "/waymo_full.txt"
waymo_full_val = destination + "/waymo_full_val.txt"
with open(waymo_full) as file:
    waymo_full_data = file.read().splitlines()
with open(waymo_full_val) as file:
    waymo_full_val_data = file.read().splitlines()

baseline = random.sample(waymo_full_data , int(len(waymo_full_data)/10) )
val2000 = waymo_full_val_data[0:-1:int(len(waymo_full_val_data)/2000)+1]
val2000 = val2000 + random.sample(waymo_full_val_data , 2000 - len(val2000) )
with open(osp.join(destination, 'waymo_baseline{}.txt'.format(len(baseline)) ), 'w') as f:
        f.writelines(f + '\n' for f in baseline)
with open(osp.join(destination, 'waymo_{}_val.txt'.format(len(val2000))), 'w') as f:
        f.writelines(f + '\n' for f in val2000)
import pdb;pdb.set_trace()
