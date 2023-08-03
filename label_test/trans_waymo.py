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
data_root       = '/SSD_DISK/users/kuangshaochen/waymo_test'
destination     = '/SSD_DISK/users/kuangshaochen/store_test_data'
data_type = 'waymo'
from label_mapping import LMS
all_num = 0
global_files = []
def write_trans_img(img_path: str):
    global all_num
    label_path = img_path.replace("_leftImg8bit.png" , "_gtFine_labelTrainIds.png").replace("img/","lab/")
    img = cv2.imread(img_path)
    label = imageio.imread(label_path)
    mylms = LMS("cityscapes","../label_test/cityscapes.json")
    mylms.read_mapping("waymo","../label_test/waymo.json")
    mylms.read_strategy("waymo2city","../label_test/waymo2city.json")
    mask = mylms.get_array()
    print(mask);quit()
    label = mask["waymo"][label]
    print(np.unique(label))
    print(mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("/SSD_DISK/users/kuangshaochen/store_test_data/gtFine/waymo/waymo{num}".format(num = all_num) + seg_map_suffix,label)
    cv2.imwrite("/SSD_DISK/users/kuangshaochen/store_test_data/leftImg/waymo/waymo{num}".format(num = all_num) + img_suffix,img)
    

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]
    paint = np.zeros(img.shape)
    for num_label in range(len(PALETTE)) :
        mask = [label == num_label]
        
        paint[mask] = np.array(PALETTE[num_label])[::-1]
        
    cv2.imwrite("/SSD_DISK/users/kuangshaochen/store_test_data/paint/waymo/waymo{num}".format(num = all_num) + seg_map_suffix, paint)
    all_num += 1
    pass
mylms = LMS("cityscapes","../label_test/cityscapes.json")
mylms.read_mapping("waymo","../label_test/waymo.json")
mylms.read_strategy("waymo2city","../label_test/waymo2city.json")
mask = mylms.get_array()
print(mask);quit()


for split in ["waymotrain"] :
    # label_card  = osp.join(data_root, "waymo_dataset" , "gtFine"        , split, "*", "*_gtFine_labelids.png")
    img_card    = osp.join(data_root, "waymo_dataset_old" , "*", "*", "img", "*_leftImg8bit.png")
    # labels = glob.glob(label_card)
    imgs = glob.glob(img_card)
    # test = imgs[0]
    # print(len(imgs))
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
        global_files.append("waymo/waymo{num}".format(num = i))
    # print(global_files)
    with open(osp.join(destination, f'{split}.txt'), 'w') as f:
        f.writelines(f + '\n' for f in global_files)
    pool.close()
    pool.join()
