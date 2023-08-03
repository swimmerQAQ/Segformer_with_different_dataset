
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
parser.add_argument('pathscene', type=str, help='example : scenexx')
args = parser.parse_args()




def write_img_lab(num , img_path: str):
    print(img_path)
    # label_path = img_path.replace("/image/","/semantic/")
    img = cv2.imread(img_path)
    # label = imageio.v2.imread(label_path)
    # print(np.unique(label))

    # cv2.imwrite(lab_write + "/{:05d}".format(num) + seg_map_suffix,label)
    cv2.imwrite(img_write + "/{:04d}".format(num) + img_suffix,img)

    pass

if __name__ == "__main__" :
    

    data_root       = args.pathscene # /SSD_DISK/users/kuangshaochen/6cam_base/nuScenes_scenes
    destination     = '/SSD_DISK/users/kuangshaochen/SegFormer/data/cityscapes/'
    allscene        = sorted(os.listdir(data_root))
    for i , scene in enumerate(allscene) :
        datas           = sorted(os.listdir(data_root + "/" + scene +'/images'))
        img_write = "/SSD_DISK/users/kuangshaochen/SegFormer/data/cityscapes/leftImg8bit/val/"+ scene
        # lab_write = "/SSD_DISK/users/kuangshaochen/SegFormer/data/cityscapes/gtFine/train/"+ scene
        filenames = []
        os.makedirs(img_write , exist_ok = True)
        # os.makedirs(lab_write , exist_ok = True)


        tqdm.write(f"Processing {len(datas)} combination files for Semantic Segmentation")
        # iterate through files
        import multiprocessing
        multi_pool =multiprocessing.Pool(processes=32)
        for i , img_path in enumerate(datas) :
            assert("{:04d}".format(i)+".png" in datas)
            _img_path = data_root+"/"+scene+'/images/' + img_path
            multi_pool.apply_async(write_img_lab,args=(i , _img_path))
            print(i)
            filenames.append(scene + "/{:04d}".format(i))
        multi_pool.close()
        multi_pool.join()
        # print(filenames)
        with open(osp.join(destination, scene + '.txt'), 'w') as f:
            f.writelines(f + '\n' for f in filenames)