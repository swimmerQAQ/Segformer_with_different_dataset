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
data_root       = '/SSD_DISK/datasets/mapillary'
destination     = '/SSD_DISK/users/kuangshaochen/SegFormer/data/cityscapes'
from label_mapping import LMS
global_files = []
def write_trans_img(num , img_path: str):
    # print(img_path , len("/SSD_DISK/datasets/mapillary/validation/v1.2/"))
    label_path = img_path.replace("/images/","/v1.2/")[:45] + "instances" + img_path[46:-4] + ".png"

    # print(label_path , len(label_path) , len("/SSD_DISK/datasets/mapillary/training/v1.2/"))
    # print(img_path)
    # print(label_path)

    img = cv2.imread(img_path)
    label = imageio.v2.imread(label_path)
    # print(np.unique(label))
    mylms = LMS("cityscapes","../label_test/cityscapes.json")
    mylms.read_mapping("mapillary","../label_test/mapillary.json")
    mylms.read_strategy("map2city","../label_test/map2city.json")
    label = np.array(label / 256, dtype=np.uint8)
    mask = mylms.get_array()
    label = mask["mapillary"][label]
    print(np.unique(label))
    

    cv2.imwrite("/SSD_DISK/users/kuangshaochen/SegFormer/data/cityscapes/gtFine/val/map/map{num}".format(num = num) + seg_map_suffix,label)
    cv2.imwrite("/SSD_DISK/users/kuangshaochen/SegFormer/data/cityscapes/leftImg8bit/val/map/map{num}".format(num = num) + img_suffix,img)


for split in ["train"] :
    img_card  = osp.join(data_root, "validation" , "images" , "*.jpg")
    label_card = osp.join(data_root, "validation" , "v1.2" , "instances" , "*.png")
    imgs = glob.glob(img_card)
    # print(imgs)
    # test = imgs[0]
    # print(test,type(test))
    # test = test.replace("/leftImg8bit/","/gtFine/")
    # test = test.replace("_leftImg8bit.png" , "_gtFine_labelids.png")
    # print(test in labels)
    # imgs = imgs[:10]
    tqdm.write(f"Processing {len(imgs)} combination files for Semantic Segmentation")
    # iterate through files
    import multiprocessing

    test=multiprocessing.Pool(processes=32)
    for i, img in enumerate(imgs) :
        print(i)
        test.apply_async(write_trans_img,args=(i , img))
    test.close()
    test.join()
    for i in range(len(imgs)) :
        global_files.append("map/map{num}".format(num = i))
    # print(global_files)
    with open(osp.join(destination, 'map.txt'), 'w') as f:
        f.writelines(f + '\n' for f in global_files)
