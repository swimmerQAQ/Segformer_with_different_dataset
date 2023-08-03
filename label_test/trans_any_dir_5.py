
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




def write_img_lab(num , img_path: str , limit_num):
    # print(img_path)
    # label_path = img_path.replace("/image/","/semantic/")
    img = cv2.imread(img_path)
    # label = imageio.v2.imread(label_path)
    # print(np.unique(label))
    # quit()
    # cv2.imwrite(lab_write + "/{:05d}".format(num) + seg_map_suffix,label)

    ################################## 
    if num >= limit_num:
        img = img[:886,:]
    cv2.imwrite(img_write + "/{:04d}".format(num) + img_suffix,img)

    pass

if __name__ == "__main__" :
    
    '''
    ['0000175', '0005145', '0014075', '0015155', '0017085', '0020075',
    '0020150', '0023120', '0026085', '0027055', '0028040', '0028130',
    '0029075', '0031030', '0031150', '0032030', '0032150', '0036100',
    '0039155', '0040100', '0042100', '0043155', '0045125', '0047140',
        '0050100', '0052080', '0053155', '0054100', '0059070', '0061160',
        '0066050', '0067130', '0069100', '0072050', '0074085', '0074170',
        '0076070', '0078150', '0079050', '0081120', '0084030', '0087070',
        '0092110', '0095050', '0097050', '0101165', '0179130', '0220150',
            '0238025', '0409175', '0453145', '0465095', '0476065', '0478115',
            '0485105', '0489125', '0508085', '0524150', '0527155', '0541125',
            '0554035', '0569140', '0570155']
    2023.2.2

    2023.2.28
    ['0158150', '0146130', '0196030', '0142100', '0124100',
     '0147030', '0162035', '0146050', '0153110', '0129170',
      '0149060', '0134035', '0100035', '0137075', '0187030',
       '0198080', '0173125', '0158030', '0133035', '0198170',
        '0160065', '0159100', '0121145', '0100160', '0012065', 
        '0139100', '0140170', '0148045', '0131170', '0199170',
         '0172120', '0122085', '0113135', '0200030', '0182155',
          '0142030', '0119045', '0105155', '0120150', '0177030',
           '0188030', '0103055', '0129125', '0197030', '0167120',
            '0131035', '0167030', '0193100', '0120035', '0174150', '0145050']
    '''
    data_root       = args.pathscene # /SSD_DISK/users/kuangshaochen/6cam_base/nuScenes_scenes
    destination     = '/HDD_DISK/datasets/data/cityscapes/'
    allscene        = os.listdir(data_root)
    # allscene = []
    # for onescene in sorted(os.listdir(data_root)) :
    #     if len(onescene) > 4 :
    #         continue
    #     if int(onescene) >= 200 and int(onescene) <= 239 :
    #         allscene.append(onescene)
    print(allscene)
    # quit()
    for i , scene in enumerate(allscene) :
        datas           = sorted(os.listdir(data_root + "/" + scene +'/images'))
        img_write = "/HDD_DISK/datasets/data/cityscapes/leftImg8bit/val/"+ scene
        # lab_write = "/HDD_DISK/datasets/data/cityscapes/gtFine/train/"+ scene
        filenames = []
        os.makedirs(img_write , exist_ok = True)
        # os.makedirs(lab_write , exist_ok = True)

        # datas = datas[:4]
        tqdm.write(f"Processing {len(datas)} combination files for Semantic Segmentation")
        # iterate through files
        import multiprocessing
        # import pdb;pdb.set_trace()
        print(scene)
        multi_pool =multiprocessing.Pool(processes=32)
        for i , img_path in enumerate(datas) :
            # print(i)
            assert("{:04d}".format(i)+".png" in datas)
            _img_path = data_root+"/"+scene+'/images/' + img_path
            multi_pool.apply_async(write_img_lab,args=(i , _img_path , len(datas)*4/5))
            # print(i)
            filenames.append(scene + "/{:04d}".format(i))
        multi_pool.close()
        multi_pool.join()
        # print(filenames)
        with open(osp.join(destination, scene + '.txt'), 'w') as f:
            f.writelines(f + '\n' for f in filenames)