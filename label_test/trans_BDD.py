import os.path as osp
import os
import cv2
import numpy as np
import imageio
img_suffix      = '_leftImg8bit.png'                #img
seg_map_suffix  = '_gtFine_labelTrainIds.png'       #label
data_root       = '/SSD_DISK/datasets/BDD'
destination     = '/SSD_DISK/users/kuangshaochen/store_test_data'
data_type = 'bdd100k'
current = osp.join(data_root, data_type)
filenames = []
for split in ['val'] :
    imgs        = os.listdir(osp.join(current, 'images/10k',split))
    img_num = 0
    for img_name in imgs :
        img_path    = osp.join( osp.join(current, 'images/10k',split),  img_name)
        label_name = img_name[:(len(img_name)-4)] + ".png"
        label_path  = osp.join( osp.join(current, 'labels/sem_seg/masks',split), label_name)
        labels = os.listdir(osp.join(current, 'labels/sem_seg/masks',split))
        assert(label_name in labels),"label 没有对应吗？ "
        img         = cv2.imread(img_path,cv2.IMREAD_COLOR)
        label       = imageio.imread(label_path)
        # print("label unique" , np.unique(label))
        img_write_path = osp.join(destination,'leftImg',data_type[:3])
        lab_write_path = osp.join(destination,'gtFine',data_type[:3])
        cv2.imwrite(img_write_path +'/'+ data_type[:3] + "{num}".format(num = img_num) + img_suffix , img)
        cv2.imwrite(lab_write_path +'/'+ data_type[:3] + "{num}".format(num = img_num) + seg_map_suffix , label)
        filename = data_type[:3]+'/'+ data_type[:3] + "{num}".format(num = img_num)
        filenames.append(filename)
        print(img_num)
        img_num += 1
    with open(osp.join(destination, f'{split}.txt'), 'w') as f:
        f.writelines(f + '\n' for f in filenames)