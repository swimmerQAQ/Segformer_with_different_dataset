
import os.path as osp
import os
import cv2
import numpy as np
import imageio
img_suffix      = '_leftImg8bit.png'                #img
seg_map_suffix  = '_gtFine_labelTrainIds.png'       #label
import argparse

parser = argparse.ArgumentParser(description='trans_any_dir')
parser.add_argument('whichscene', type=str, help='example : scenexx')
args = parser.parse_args()
data_root       = "/SSD_DISK/users/kuangshaochen/6cam_"+ args.whichscene +"_revision"
destination     = '/SSD_DISK/users/kuangshaochen/SegFormer/data/cityscapes/'
datas           = os.listdir(data_root+'/images')
write = "/SSD_DISK/users/kuangshaochen/SegFormer/data/cityscapes/leftImg8bit/val/"+ args.whichscene
filenames = []
os.mkdir(write)
for i in range(len(datas)) :
    name = write + "/{:04d}".format(i)
    assert("{:04d}".format(i)+".png" in datas)
    
    img =cv2.imread(data_root+"/images/""{:04d}".format(i)+".png")
    print(name + img_suffix)
    cv2.imwrite(name + img_suffix,img)
    filenames.append(args.whichscene +"/{:04d}".format(i))
with open(osp.join(destination, args.whichscene + '.txt'), 'w') as f:
    f.writelines(f + '\n' for f in filenames)