
import os.path as osp
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import imageio
img_suffix      = '_leftImg8bit.png'                #img
seg_map_suffix  = '_gtFine_labelTrainIds.png'       #label
parser = argparse.ArgumentParser(description='trans_any_dir')
parser.add_argument('whichscene', type=str, help='example : scenexx')
args = parser.parse_args()
data_root       = "/HDD_DISK/datasets/data/liwenye/"+ args.whichscene
destination     = '/HDD_DISK/datasets/data/cityscapes/'
datas           = sorted(os.listdir(data_root+'/image'))
img_write = "/HDD_DISK/datasets/data/cityscapes/leftImg8bit/train/"+ args.whichscene
lab_write = "/HDD_DISK/datasets/data/cityscapes/gtFine/train/"+ args.whichscene
filenames = []


def write_img_lab(num , img_path: str):
    print(img_path)
    label_path = img_path.replace("/image/","/semantic/")
    img = cv2.imread(img_path)
    label = imageio.v2.imread(label_path)
    print(np.unique(label) , img_path.split('/')[-1][:5])

    # cv2.imwrite(lab_write + "/{:05d}".format(num) + seg_map_suffix,label)
    # cv2.imwrite(img_write + "/{:05d}".format(num) + img_suffix,img)
    cv2.imwrite(lab_write +"/"+ img_path.split('/')[-1][:5] + seg_map_suffix,label)
    cv2.imwrite(img_write +"/"+ img_path.split('/')[-1][:5] + img_suffix,img)

    pass

if __name__ == "__main__" :
    
    os.makedirs(img_write , exist_ok = True)
    os.makedirs(lab_write , exist_ok = True)


    tqdm.write(f"Processing {len(datas)} combination files for Semantic Segmentation")
    # iterate through files
    import multiprocessing
    multi_pool =multiprocessing.Pool(processes=32)
    for i , img_path in enumerate(datas) :
        # assert("{:05d}".format(i)+".png" in datas)
        _img_path = data_root+'/image/' + img_path
        # multi_pool.apply_async(write_img_lab,args=(i , _img_path))
        print(i)
        filenames.append(args.whichscene +"/"+ img_path.split('/')[-1][:5])
    multi_pool.close()
    multi_pool.join()
    print(filenames)
    with open(osp.join(destination, args.whichscene + '.txt'), 'w') as f:
        f.writelines(f + '\n' for f in filenames)