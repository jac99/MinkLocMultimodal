# Author: Jacek Komorowski
# Warsaw University of Technology

# Dataset wrapper for Oxford laser scans dataset from PointNetVLAD project
# For information on dataset see: https://github.com/mikacuy/pointnetvlad

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import psutil
from bitarray import bitarray
import tqdm
from datasets.augmentation import TrainRGBTransform, ValRGBTransform, tensor2img
from PIL import Image
import random

DEBUG = False


class OxfordDataset(Dataset):
    """
    Dataset wrapper for Oxford laser scans dataset from PointNetVLAD project.
    """
    def __init__(self, dataset_path, query_filename, image_path=None, lidar2image_ndx=None,
                 transform=None, set_transform=None, image_transform=None, max_elems=None, use_cloud=True):
        # transform: transform applied to each element
        # set transform: transform applied to the entire set (anchor+positives+negatives); the same transform is applied
        if DEBUG:
            print('Initializing dataset: {}'.format(dataset_path))
            print(psutil.virtual_memory())
        assert os.path.exists(dataset_path), 'Cannot access dataset path: {}'.format(dataset_path)
        self.dataset_path = dataset_path
        self.query_filepath = os.path.join(dataset_path, query_filename)
        assert os.path.exists(self.query_filepath), 'Cannot access query file: {}'.format(self.query_filepath)
        self.transform = transform
        self.set_transform = set_transform
        self.image_path = image_path
        self.lidar2image_ndx = lidar2image_ndx
        self.image_transform = image_transform
        self.max_elems = max_elems
        self.n_points = 4096    # pointclouds in the dataset are downsampled to 4096 points
        self.image_ext = '.png'
        self.use_cloud = use_cloud

        cached_query_filepath = os.path.splitext(self.query_filepath)[0] + '_cached.pickle'
        if not os.path.exists(cached_query_filepath):
            # Pre-process query file
            self.queries = self.preprocess_queries(self.query_filepath, cached_query_filepath)
        else:
            print('Loading preprocessed query file: {}...'.format(cached_query_filepath))
            with open(cached_query_filepath, 'rb') as handle:
                # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
                self.queries = pickle.load(handle)

        if max_elems is not None:
            filtered_queries = {}
            for ndx in self.queries:
                if ndx >= self.max_elems:
                    break
                filtered_queries[ndx] = {'query': self.queries[ndx]['query'],
                                         'positives': self.queries[ndx]['positives'][0:max_elems],
                                         'negatives': self.queries[ndx]['negatives'][0:max_elems]}
            self.queries = filtered_queries

        print('{} queries in the dataset'.format(len(self)))

    def preprocess_queries(self, query_filepath, cached_query_filepath):
        print('Loading query file: {}...'.format(query_filepath))
        with open(query_filepath, 'rb') as handle:
            # key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
            queries = pickle.load(handle)

        # Convert to bitarray
        for ndx in tqdm.tqdm(queries):
            queries[ndx]['positives'] = set(queries[ndx]['positives'])
            queries[ndx]['negatives'] = set(queries[ndx]['negatives'])
            pos_mask = [e_ndx in queries[ndx]['positives'] for e_ndx in range(len(queries))]
            neg_mask = [e_ndx in queries[ndx]['negatives'] for e_ndx in range(len(queries))]
            queries[ndx]['positives'] = bitarray(pos_mask)
            queries[ndx]['negatives'] = bitarray(neg_mask)

        with open(cached_query_filepath, 'wb') as handle:
            pickle.dump(queries, handle)

        return queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ndx):
        # Load point cloud and apply transform
        filename = self.queries[ndx]['query']
        result = {'ndx': ndx}
        if self.use_cloud:
            query_pc = self.load_pc(filename)
            # Apply transformations
            if self.transform is not None:
                query_pc = self.transform(query_pc)

            result['cloud'] = query_pc

        if self.image_path is not None:
            img = image4lidar(filename, self.image_path, self.image_ext, self.lidar2image_ndx, k=None)
            if self.image_transform is not None:
                img = self.image_transform(img)
            result['image'] = img

        return result

    def get_item_by_filename(self, filename):
        # Load point cloud and apply transform
        query_pc = self.load_pc(filename)
        if self.transform is not None:
            query_pc = self.transform(query_pc)
        return query_pc

    def get_items(self, ndx_l):
        # Load multiple point clouds and stack into (batch_size, n_points, 3) tensor
        clouds = [self[ndx][0] for ndx in ndx_l]
        clouds = torch.stack(clouds, dim=0)
        return clouds

    def get_positives_ndx(self, ndx):
        # Get list of indexes of similar clouds
        return self.queries[ndx]['positives'].search(bitarray([True]))

    def get_negatives_ndx(self, ndx):
        # Get list of indexes of dissimilar clouds
        return self.queries[ndx]['negatives'].search(bitarray([True]))

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
    #assert os.path.exists(image_file_path), 'Cannot find image: {}'.format(image_file_path)

    img = Image.open(image_file_path)
    return img


if __name__ == '__main__':
    dataset_path = '/media/sf_Datasets/PointNetVLAD'
    query_filename = 'test_queries_baseline.pickle'
    images_path = '/media/sf_Datasets/images4lidar'
    lidar2image_ndx_path = '/media/sf_Datasets/images4lidar/lidar2image_ndx.pickle'
    lidar2image_ndx = pickle.load(open(lidar2image_ndx_path, 'rb'))

    train_transform = TrainRGBTransform(aug_mode=3)
    val_transform = ValRGBTransform()

    ds = OxfordDataset(dataset_path, query_filename, images_path, lidar2image_ndx, image_transform=train_transform)
    for i in range(20):
        e = ds[261]
        img = tensor2img(e['image'])
        img.save("tmp{}.png".format(i))
    print('.')
