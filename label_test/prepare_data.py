import os
from PIL import Image
import numpy as np
from label_mapping import LMS
class map_data:
    def __init__(self):
        self.all_imgs = {}
        self.all_labels = {}
        pass
    def process_mapillary(self,relative_path2mapillary):
        dirs = os.listdir(relative_path2mapillary)
        for dir_name in dirs :
            if dir_name == "training" or dir_name == "validation" :
                if os.path.isdir(os.path.join(relative_path2mapillary,dir_name)) :
                    imgs_path = os.path.join(relative_path2mapillary,dir_name, "images")
                    labels_path = os.path.join(relative_path2mapillary,dir_name, "v1.2","instances")

                    imgs = os.listdir(imgs_path)
                    labels = os.listdir(labels_path)

                    num = 0
                    mylms = LMS("cityscapes","./cityscapes.json")
                    mylms.read_mapping("mapillary","./mapillary.json")
                    mylms.read_strategy("map2city","./map2city.json")
                    mask = mylms.get_array()

                    for img in imgs:
                        assert(img[:22]+".png" in labels) , "大哥，不在里面吧？"
                        image_name = "img_map_{num}".format(num = num) + ".png"
                        label_name = "lab_map_{num}".format(num = num) + ".png"
                        
                        img_path = os.path.join(relative_path2mapillary, dir_name, "images",img)
                        label_path = os.path.join(relative_path2mapillary, dir_name, "v1.2","instances",img[:22]+".png")

                        img = Image.open(img_path)
                        img_array = np.array(img)

                        label = Image.open(label_path)
                        lab_np = np.array(label)
                        label_array = np.array(lab_np / 256, dtype=np.uint8)
                        print("\nmapping: ",np.unique(label_array))

                        img_mapped = img_array
                        label_mapped = mask["mapillary"][label_array]
                        print("mapped: ",np.unique(label_mapped))

                        data_type = "mapillary"
                        relative_hybrid_dataset_path = os.path.join("../hybrid_dataset", data_type)
                        assert(os.path.isdir(relative_hybrid_dataset_path) == True),"没有hybrid data文件夹，啊这？"
                        assert(os.path.isdir(relative_hybrid_dataset_path+"/train") == True),"没有train文件夹，啊这？"
                        assert(os.path.isdir(relative_hybrid_dataset_path+"/val") == True),"没有val文件夹，啊这？"
                        if dir_name == "training" :
                            store_path = "../hybrid_dataset/mapillary/train"
                        elif dir_name == "validation" :
                            store_path = "../hybrid_dataset/mapillary/val"
                        img_from_PIL = Image.fromarray(img_mapped)
                        label_from_PIL = Image.fromarray(label_mapped)
                        print(label_mapped.shape)
                        img_from_PIL.save(store_path + "/images/" + image_name)
                        label_from_PIL.save(store_path + "/labels/" + label_name)
                        num += 1 

    def test_for_process_mapillary(self,image_path):
        img = Image.open(image_path)
        img_np = np.array(img)
        instance_label_array = np.array(img_np / 256, dtype=np.uint8)
        mylms = LMS("cityscapes","./cityscapes.json")
        mylms.read_mapping("mapillary","./mapillary.json")
        mylms.read_strategy("map2city","./map2city.json")
        mask = mylms.get_array()
        # mask["mapillary"][8] = 255
        # print(mask["mapillary"][30])
        # print(img_np.size)
        instance_label_array = mask["mapillary"][instance_label_array]
        # print(instance_label_array.size)
        label_test = Image.fromarray(instance_label_array)
        label_test.save("label_have_.jpg")

    def test_city_process(self,relative_path2cityscapes):
        dirs = os.listdir(relative_path2cityscapes)
        for dir_name in dirs :
            if dir_name == "leftImg8bit" :
                if os.path.isdir(os.path.join(relative_path2cityscapes,dir_name)) :
                    for split in ["val"]:
                        train_path = os.path.join(relative_path2cityscapes,dir_name, split)
                        # labels_path = os.path.join(relative_path2cityscapes,"gtFine","train")
                        img_types = os.listdir(train_path)
                        # label_types = os.listdir(labels_path)
                        num = 0
                        for img_type in img_types :
                            print("区域类型：",img_type)
                            img_path = os.path.join(train_path,img_type)
                            imgs = os.listdir(img_path)
                            label_path = os.path.join(relative_path2cityscapes,"gtFine",split,img_type)
                            labels = os.listdir(label_path)

                            label_valids = list()
                            for label in labels :
                                valid_num = len(label) - 12
                                if label[valid_num:] == "TrainIds.png" :
                                    label_valids.append(label)
                        
                            for img in imgs :
                                valid_num = len(img) - 15
                                assert(img[:valid_num]+"gtFine_labelTrainIds.png" in labels) , "大哥，不在里面吧？"
                                image_name = "img_cit_{num}".format(num = num) + ".png"
                                label_name = "lab_cit_{num}".format(num = num) + ".png"
                            
                                store_img_path = os.path.join(img_path,img)
                                store_label_path = os.path.join(label_path,img[:valid_num]+"gtFine_labelTrainIds.png")

                                img = Image.open(store_img_path)

                                label = Image.open(store_label_path)
                                # print(store_label_path)
                                store_path = "../hybrid_dataset/cityscapes/" + split
                                img.save(store_path + "/images/" + image_name)
                                label.save(store_path + "/labels/" + label_name)
                                num += 1
                                print("当前：",num," ",split)

myc = map_data()
myc.process_mapillary("../../../datasets/mapillary")

# image_path = "/SSD_DISK/datasets/mapillary/training/v1.2/instances/__CRyFzoDOXn6unQ6a3DnQ.png"
# myc.test_city_process("../../../datasets/cityscapes")

# from label_mapping import LMS
# mylms = LMS("cityscapes","./cityscapes.json")
# mylms.read_mapping("mapillary","./mapillary.json")
# mylms.read_strategy("map2city","./map2city.json")
# mylms.show_remapping()
# mylms.map_action()
# mylms.show_array()