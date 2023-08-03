import os
path = "/SSD_DISK/users/kuangshaochen/SegFormer/"
import glob
delete_files = glob.glob(os.path.join(path , "*.npy"))
print(delete_files , os.path.join(path , "*.npy"))
for delete_file in delete_files : 
    # print(delete_file)
    os.remove(delete_file)
import numpy as np
# for read in delete_files:
#     temp = np.load(read)
#     np.save(path + "/type19/" + read[-8:] , temp)
