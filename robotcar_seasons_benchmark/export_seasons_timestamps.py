"""
Export list of timestamps for all images in the RobotCar Seasons dataset.
Timestamps from each traversal are saved in the separated .csv file.
"""

import numpy as np

from robotcar_seasons_benchmark.robotcar_seasons import RobotCarSeasonsDataset
from robotcar_seasons_benchmark.robotcar import RobotCarDataset


if __name__ == '__main__':
    seasons_dataset_root = '/media/sf_Datasets/robotcar-seasons'
    seasons_ds = RobotCarSeasonsDataset(seasons_dataset_root)
    robotcar_root = '/media/sf_Datasets/RobotCar/'
    robotcar_ds = RobotCarDataset(robotcar_root)

    images_traversal_ndx = {}
    # Split image timestamps into traversals
    for image_ts in seasons_ds.index_ts_image:
        traversal = robotcar_ds.get_traversal(image_ts)
        scan_ts, _ = robotcar_ds.find_closest_scan(image_ts)
        scan_ts = np.int64(scan_ts)
        if traversal not in images_traversal_ndx:
            images_traversal_ndx[traversal] = []
        images_traversal_ndx[traversal].append((image_ts, scan_ts))

    for traversal in images_traversal_ndx:
        with open("seasons_" + traversal + ".csv", "w") as file:
            for e in images_traversal_ndx[traversal]:
                line = '{}, {}\n'.format(e[0], e[1])
                file.write(line)
    print('RobotCar Season image timestamps split into traversal.')
