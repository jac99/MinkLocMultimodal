import os
import pickle
from bisect import bisect_left
import numpy as np
import pathlib

DEBUG = False


class RobotCarDataset:
    # Oxford RobotCar dataset
    def __init__(self, dataset_root, lidar_subfolder='lms_front', lidar_timestamp=''):
        print('Processing Oxford RobotCar dataset: {}'.format(dataset_root))
        assert os.path.exists(dataset_root), 'Cannot find RobotCar dataset: {}'.format(dataset_root)
        self.dataset_root = dataset_root
        self.lidar_subfolder = lidar_subfolder
        self.lidar_timestamps_file = self.lidar_subfolder + '.timestamps'
        self.ts_len = 16    # Timestamp length
        self.lidar_file_ext = '.bin'

        traversals = [os.path.join(self.dataset_root, e) for e in os.listdir(self.dataset_root)]
        traversals = [e for e in traversals if os.path.isdir(e)]
        print('{} traversals found'.format(len(traversals)))

        self.lidar_ts_ndx = {}  # lidar_ts_nds[ts] = (lidar_file, traversal_name, chunk)

        root_path = pathlib.Path(__file__).parent
        cached_indexes_file = os.path.join(root_path, 'cached_indexes.pickle')
        if not os.path.exists(cached_indexes_file):
            for traversal in traversals:
                self.index_lidar_scans(traversal)
                if DEBUG:
                    break

            pickle.dump(self.lidar_ts_ndx, open(cached_indexes_file, "wb"))
        else:
            self.lidar_ts_ndx = pickle.load(open(cached_indexes_file, "rb"))
            print('Cached index of LiDAR scans loaded from: {}'.format(cached_indexes_file))

        print('{} LiDAR scans indexed'.format(len(self.lidar_ts_ndx)))
        self.sorted_ts_list = [np.int64(e) for e in list(self.lidar_ts_ndx)]
        self.sorted_ts_list.sort()
        print('')

    def index_lidar_scans(self, traversal):
        print('Indexing LiDAR scans in: {}...'.format(traversal), end='')
        lidar_path = os.path.join(traversal, self.lidar_subfolder)
        if not os.path.exists(lidar_path):
            print('WARNING: Cannot find LiDAR data: {}'.format(lidar_path))
            return

        lidar_timestamp_filepath = os.path.join(traversal, self.lidar_timestamps_file)
        if not os.path.exists(lidar_timestamp_filepath):
            print('WARNING: Cannot find LiDAR timestamps file: {}'.format(lidar_timestamp_filepath))
            return

        # Process LiDAR timestamp file
        with open(lidar_timestamp_filepath, 'r') as file:
            content = file.readlines()
            content = [x.rstrip('\n') for x in content]

        count_scans = 0
        for line in content:
            assert line[self.ts_len] == ' ', 'Incorrect line in LiDAR timestamp file: {}'.format(line)
            ts = line[:self.ts_len]
            chunk = line[self.ts_len + 1:]
            lidar_file_path = os.path.join(traversal, self.lidar_subfolder, ts + self.lidar_file_ext)
            if not os.path.exists(lidar_file_path):
                print('WARNING: Cannot find LiDAR file: {}'.format(lidar_file_path))
                break
            # lidar_ts_nds[ts] = (lidar_file, traversal_name, chunk)
            self.lidar_ts_ndx[ts] = (lidar_file_path, os.path.split(traversal)[1], chunk)
            count_scans += 1

        print(' {} scans found'.format(count_scans))

    def find_closest_scan(self, ts):
        # Find the closest scan given the timestamp
        ts = np.int64(ts)   # timestamps are indexed as int64 values
        pos = bisect_left(self.sorted_ts_list, ts)
        if pos == 0:
            closest_scan_ts = self.sorted_ts_list[0]
        elif pos == len(self.sorted_ts_list):
            closest_scan_ts = self.sorted_ts_list[-1]
        else:
            before = self.sorted_ts_list[pos - 1]
            after = self.sorted_ts_list[pos]
            closest_scan_ts = after if after - ts < ts - before else before

        delta = closest_scan_ts - ts if closest_scan_ts > ts else ts - closest_scan_ts
        # Convert back to string
        return str(closest_scan_ts), delta

    def get_traversal(self, ts, threshold=100):
        # Get traversal for the timestamp
        # threshold in ms
        scan_ts, delta = self.find_closest_scan(ts)
        assert delta < threshold*1000, 'Cannot find scan within the threshold. delta={}'.format(delta)
        lidar_file, traversal, chunk = self.lidar_ts_ndx[scan_ts]
        return traversal


if __name__ == '__main__':
    root = '/media/sf_Datasets/RobotCar/'
    ds = RobotCarDataset(root)
    ts = '324324234234'
    closest_scan_ts, delta = ds.find_closest_scan(ts)
    print(ds.lidar_ts_ndx[closest_scan_ts], delta)

