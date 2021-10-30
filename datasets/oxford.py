# Author: Jacek Komorowski
# Warsaw University of Technology

# Dataset wrapper for Oxford laser scans dataset from PointNetVLAD project
# For information on dataset see: https://github.com/mikacuy/pointnetvlad

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from typing import Dict

DEBUG = False


class OxfordDataset(Dataset):
    """
    Dataset wrapper for Oxford laser scans dataset from PointNetVLAD project.
    """
    def __init__(self, dataset_path: str, query_filename: str, image_path: str = None,
                 lidar2image_ndx_path: str = None, transform=None, set_transform=None, image_transform=None,
                 use_cloud: bool = True):
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform
        self.queries: Dict[int, TrainingTuple] = pickle.load(open(self.query_filepath, 'rb'))
        self.image_path = image_path
        self.lidar2image_ndx_path = lidar2image_ndx_path
        self.image_transform = image_transform
        self.n_points = 4096    # pointclouds in the dataset are downsampled to 4096 points
        self.image_ext = '.png'
        self.use_cloud = use_cloud
        print('{} queries in the dataset'.format(len(self)))

        assert os.path.exists(self.lidar2image_ndx_path), f"Cannot access lidar2image_ndx: {self.lidar2image_ndx_path}"
        self.lidar2image_ndx = pickle.load(open(self.lidar2image_ndx_path, 'rb'))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        filename = self.queries[ndx].rel_scan_filepath
        result = {'ndx': ndx}
        if self.use_cloud:
            # Load point cloud and apply transform
            file_pathname = os.path.join(self.dataset_path, self.queries[ndx].rel_scan_filepath)
            query_pc = self.load_pc(file_pathname)
            if self.transform is not None:
                query_pc = self.transform(query_pc)
            result['cloud'] = query_pc

        if self.image_path is not None:
            img = image4lidar(filename, self.image_path, self.image_ext, self.lidar2image_ndx, k=None)
            if self.image_transform is not None:
                img = self.image_transform(img)
            result['image'] = img

        return result

    def get_positives(self, ndx):
        return self.queries[ndx].positives

    def get_non_negatives(self, ndx):
        return self.queries[ndx].non_negatives

    def load_pc(self, filename):
        # Load point cloud, does not apply any transform
        # Returns Nx3 matrix
        file_path = os.path.join(self.dataset_path, filename)
        pc = np.fromfile(file_path, dtype=np.float64)
        # coords are within -1..1 range in each dimension
        assert pc.shape[0] == self.n_points * 3, "Error in point cloud shape: {}".format(file_path)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        pc = torch.tensor(pc, dtype=torch.float)
        return pc


def ts_from_filename(filename):
    # Extract timestamp (as integer) from the file path/name
    temp = os.path.split(filename)[1]
    lidar_ts = os.path.splitext(temp)[0]        # LiDAR timestamp
    assert lidar_ts.isdigit(), 'Incorrect lidar timestamp: {}'.format(lidar_ts)

    temp = os.path.split(filename)[0]
    temp = os.path.split(temp)[0]
    traversal = os.path.split(temp)[1]
    assert len(traversal) == 19, 'Incorrect traversal name: {}'.format(traversal)

    return int(lidar_ts), traversal


def image4lidar(filename, image_path, image_ext, lidar2image_ndx, k=None):
    # Return an image corresponding to the given lidar point cloud (given as a path to .bin file)
    # k: Number of closest images to randomly select from
    lidar_ts, traversal = ts_from_filename(filename)
    assert lidar_ts in lidar2image_ndx, 'Unknown lidar timestamp: {}'.format(lidar_ts)

    # Randomly select one of images linked with the point cloud
    if k is None or k > len(lidar2image_ndx[lidar_ts]):
        k = len(lidar2image_ndx[lidar_ts])

    image_ts = random.choice(lidar2image_ndx[lidar_ts][:k])
    image_file_path = os.path.join(image_path, traversal, str(image_ts) + image_ext)
    #image_file_path = '/media/sf_Datasets/images4lidar/2014-05-19-13-20-57/1400505893134088.png'
    img = Image.open(image_file_path)
    return img


class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # position: x, y position in meters (northing, easting)
        assert position.shape == (2,)

        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position
