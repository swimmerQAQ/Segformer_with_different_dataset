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
data_root       = '/SSD_DISK/datasets/IDD'
destination     = '/SSD_DISK/users/kuangshaochen/store_test_data'
data_type = 'idd_se_seg'
from label_mapping import LMS
all_num = 0
global_files = []
def write_trans_img(img_path: str):
    global all_num
    label_path = img_path.replace("/leftImg8bit/","/gtFine/").replace("_leftImg8bit.png" , "_gtFine_labelids.png").replace("_leftImg8bit.jpg" , "_gtFine_labelids.png")
    img = cv2.imread(img_path)
    label = imageio.imread(label_path)
    mylms = LMS("cityscapes","../label_test/cityscapes.json")
    mylms.read_mapping("idd","../label_test/idd.json")
    mylms.read_strategy("idd2city","../label_test/idd2city.json")
    mask = mylms.get_array()
    label = mask["idd"][label]
    # print(np.unique(label))
    # print(all_num)

    cv2.imwrite("/SSD_DISK/users/kuangshaochen/store_test_data/gtFine/idd/idd{num}".format(num = all_num) + seg_map_suffix,label)
    cv2.imwrite("/SSD_DISK/users/kuangshaochen/store_test_data/leftImg/idd/idd{num}".format(num = all_num) + img_suffix,img)
    all_num += 1
    pass



for split in ["val"] :
    label_card  = osp.join(data_root, "*" , "gtFine"        , split, "*", "*_gtFine_labelids.png")
    img_card    = osp.join(data_root, "*" , "leftImg8bit"   , split, "*", "*_leftImg8bit.*")
    labels = glob.glob(label_card)
    imgs = glob.glob(img_card)
    # test = imgs[0]
    # print(test,type(test))
    # test = test.replace("/leftImg8bit/","/gtFine/")
    # test = test.replace("_leftImg8bit.png" , "_gtFine_labelids.png")
    # print(test in labels)
    # imgs = imgs[:10]
    tqdm.write(f"Processing {len(imgs)} combination files for Semantic Segmentation")
    # iterate through files
    process = 0
    tqdm.write("Progress: {:>3} %".format(process * 100 / len(imgs)), end=" ")
    pool = Pool(1)
    results = list(tqdm(pool.imap(write_trans_img, imgs),total = len(imgs)))
    for i in range(len(imgs)) :
        global_files.append("idd/idd{num}".format(num = i))
    # print(global_files)
    with open(osp.join(destination, f'{split}.txt'), 'w') as f:
        f.writelines(f + '\n' for f in global_files)
    pool.close()
    pool.join()
