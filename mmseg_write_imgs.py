import argparse
import os
import imageio
import cv2
import numpy as np
from label_mapping import LMS
def parse_args():
    parser = argparse.ArgumentParser(
        description=' load dir_imgs(RGB) to mcity34ement ')
    parser.add_argument('ImgPath', help=' path to imgs. like: images: xxx1.png, xxx2.png,.....,xxxx.png ')
    args = parser.parse_args()
    return args

def main():
    img_suffix      = '_leftImg8bit.png'                #img
    seg_map_suffix  = '_gtFine_labelTrainIds.png'       #label
    args = parse_args()
    print(args.ImgPath)
    ImgNames = os.listdir(args.ImgPath)

    filenames = []
    for ImgName in ImgNames :
        img_path = os.path.join(args.ImgPath, ImgName)
        # img = cv2.imread(img_path)
        # img = imageio.imread(img_path)
        img = cv2.imread(img_path)
        # num = ImgName[:9]
        print("write", ImgName," ", ImgName[:-4])
        
        # print("pre lab", np.unique(img))
        ##################################3
        mylms = LMS("cityscapes","./label_test/cityscapes.json")
        mylms.read_mapping("city34","./label_test/city34.json")
        mylms.read_strategy("city342city","./label_test/city342city.json")
        mask = mylms.get_array()
        img = mask["city34"][img]
        print("unique lab", np.unique(img))
        ##################################
        cv2.imwrite("/SSD_DISK/users/kuangshaochen/store_test_data/leftImg/nuscenes/nuscenes"+ ImgName[:-4] +img_suffix,img)
        filenames.append("nuscenes/nuscenes" + ImgName[:-4])
    with open(os.path.join("/SSD_DISK/users/kuangshaochen/store_test_data/nuscenesval.txt"), 'w') as f:
        f.writelines(f + '\n' for f in filenames)


if __name__ == '__main__':
    main()