import os.path as osp
import os
import cv2
import numpy as np
import imageio
import glob
from multiprocessing import Pool
from tqdm import tqdm
img_suffix      = '_leftImg8bit.png'                #img
seg_map_suffix  = '_gtFine_labelTrainIds.png'       #label
data_root       = '../img_and_seg/'
destination     = '/SSD_DISK/users/kuangshaochen/SegFormer/data/cityscapes'
labels = np.load(data_root + "semantic/semantic.npy")
from label_mapping import LMS
global_files = []
def write_trans_img(num , img_path: str):
    img = cv2.imread(img_path)
    label = labels[num]
    # print(np.unique(label))
    mylms = LMS("cityscapes","../label_test/cityscapes.json")
    mylms.read_mapping("waymo","../label_test/waymo.json")
    mylms.read_strategy("waymo2city","../label_test/waymo2city.json")
    mask = mylms.get_array()
    print(mask)
    label = mask["waymo"][label]
    print(np.unique(label))
    
    cv2.imwrite("/SSD_DISK/users/kuangshaochen/SegFormer/data/cityscapes/gtFine/train/nerf/nerf{num}".format(num = num) + seg_map_suffix,label)
    cv2.imwrite("/SSD_DISK/users/kuangshaochen/SegFormer/data/cityscapes/leftImg8bit/train/nerf/nerf{num}".format(num = num) + img_suffix,img)


for split in ["train"] :
    img_card  = osp.join(data_root, "new_image" , "*.png")
    imgs = glob.glob(img_card)
    os.makedirs("/SSD_DISK/users/kuangshaochen/SegFormer/data/cityscapes/gtFine/train/nerf", exist_ok=True)
    os.makedirs("/SSD_DISK/users/kuangshaochen/SegFormer/data/cityscapes/leftImg8bit/train/nerf", exist_ok=True)
    tqdm.write(f"Processing {len(imgs)} combination files for Semantic Segmentation")
    # iterate through files
    import multiprocessing
    # imgs = imgs[:2]
    test=multiprocessing.Pool(processes=32)
    for i in range(len(imgs)) :
        img = "../img_and_seg/new_image/{:05d}.png".format(i)
        test.apply_async(write_trans_img,args=(i , img))
    test.close()
    test.join()
    for i in range(len(imgs)) :
        global_files.append("nerf/nerf{num}".format(num = i))
    # print(global_files)
    with open(osp.join(destination, 'nerf.txt'), 'w') as f:
        f.writelines(f + '\n' for f in global_files)
