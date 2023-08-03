from importlib.resources import path
import numpy as np
from nuscenes import NuScenes
data_path = "/SSD_DISK/datasets/nuscenes"
nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)
# info = nusc_infos[index]
# lidar_path = info['lidar_path'][16:]
# lidar_sd_token = nusc.get('sample', info['token'])['data']['LIDAR_TOP']
# lidarseg_labels_filename = os.path.join(nusc.dataroot,
#                                                 nusc.get('lidarseg', lidar_sd_token)['filename'])

# points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
# points_label = np.vectorize(learning_map.__getitem__)(points_label)
# points = np.fromfile(os.path.join(data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

# data_tuple = (points[:, :3], points_label.astype(np.uint8))#points 0 1 2