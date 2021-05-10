"""
Computes mapping between RobotCar Seasons image path and a corresponding point cloud path
"""

import os
import pickle
import tqdm

from robotcar_seasons_benchmark.robotcar_seasons import RobotCarSeasonsDataset
from robotcar_seasons_benchmark.robotcar import RobotCarDataset


if __name__ == '__main__':
    seasons_dataset_root = '/media/sf_Datasets/robotcar-seasons'
    seasons_ds = RobotCarSeasonsDataset(seasons_dataset_root)
    robotcar_root = '/media/sf_Datasets/RobotCar/'
    robotcar_ds = RobotCarDataset(robotcar_root)

    pointcloud_l = []
    count_missing_clouds = 0
    # Split image timestamps into traversals
    for image_ts in tqdm.tqdm(seasons_ds.index_ts_rel_image):
        traversal = robotcar_ds.get_traversal(image_ts)
        rel_image_path = seasons_ds.index_ts_rel_image[image_ts]
        rel_cloud_path = os.path.join(traversal, 'pointclouds', str(image_ts) + '.bin')
        abs_cloud_path = os.path.join(robotcar_root, rel_cloud_path)
        if not os.path.exists(abs_cloud_path):
            print('Missing point cloud: {}'.format(abs_cloud_path))
            count_missing_clouds += 1
            continue

        pointcloud_l.append((image_ts, rel_image_path, rel_cloud_path))

    print('{} missing point clouds'.format(count_missing_clouds))
    pickle.dump(pointcloud_l, open("season_scans.pickle", "wb"))
