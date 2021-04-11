# For each LiDAR scan in the dataset find the corresponding RGB image based on the timestamp

import os
import tqdm
import pickle
import argparse
import numpy as np
from PIL import Image
from multiprocessing import Pool

from misc.utils import MinkLocParams
from datasets.dataset_utils import make_datasets
from thirdparty.robotcardatasetsdk.image import load_image


def index_readings(root_folder, sensor_folder, ext='.jpg', cache_path='.'):
    # Index all RGB  images in the root folder
    print('Indexing root folder: {}'.format(root_folder))

    assert os.path.exists(root_folder), 'Cannot access root folder: {}'.format(root_folder)

    tmp = sensor_folder.replace('/', '_')   # Fix for stereo subfolders
    cache_filepath = os.path.join(cache_path, 'cached_ndx_{}.pickle'.format(tmp))
    if os.path.exists(cache_filepath):
        print('Using cached index: {}'.format(cache_filepath))
        with open(cache_filepath, 'rb') as f:
            reading_ts_ndx = pickle.load(f)
    else:
        reading_ts_ndx = {}

        # Index all traversals
        traversals = [e for e in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, e))]
        for t in tqdm.tqdm(traversals):
            traversal_path = os.path.join(root_folder, t)
            readings_folder = os.path.join(traversal_path, sensor_folder)
            if not os.path.exists(readings_folder):
                continue
            images = [e for e in os.listdir(readings_folder) if os.path.isfile(os.path.join(readings_folder, e)) and
                      os.path.splitext(e)[1] == ext]
            for img in images:
                img_filepath = os.path.join(readings_folder, img)
                img_ts = os.path.splitext(img)[0]
                reading_ts_ndx[img_ts] = (img_filepath, t)

        with open(cache_filepath, 'wb') as handle:
            pickle.dump(reading_ts_ndx, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('{} readings in: {} - {}'.format(len(reading_ts_ndx), root_folder, sensor_folder))
    return reading_ts_ndx


def index_scans_in_datasets(params: MinkLocParams):

    scans_ts_ndx = {}

    # Training and validation dataset
    datasets = make_datasets(params, debug=False)

    for ds in datasets:
        print('Indexing LiDAR scans in dataset {}: {}'.format(ds, datasets[ds].dataset_path))
        for e in datasets[ds].queries:
            ts, traversal = get_ts_traversal(datasets[ds].queries[e]['query'])
            scans_ts_ndx[ts] = traversal

    # Evaluation datasets
    for database_file, query_file in zip(params.eval_database_files, params.eval_query_files):
        # Extract location name from query and database files
        location_name = database_file.split('_')[0]
        if location_name != 'oxford':
            continue

        temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        p = os.path.join(params.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(params.dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        for set in database_sets:
            tmp_ndx = index_eval_set(set)
            scans_ts_ndx.update(tmp_ndx)

        for set in query_sets:
            tmp_ndx = index_eval_set(set)
            scans_ts_ndx.update(tmp_ndx)

    print('Indexed {} LiDAR point cloud'.format(len(scans_ts_ndx)))
    return scans_ts_ndx


def get_ts_traversal(query_relative_path):
    temp1, temp2 = os.path.split(query_relative_path)
    temp1 = os.path.split(temp1)[0]
    traversal = os.path.split(temp1)[1]
    ts = os.path.splitext(temp2)[0]
    assert ts.isdigit(), 'Incorrect timestamp: {}'.format(ts)
    assert 16 <= len(ts) <= 16, 'Incorrect timestamp: {}'.format(ts)
    assert len(traversal) == 19, 'Incorrect traversal: {}'.format(traversal)
    return ts, traversal


def index_eval_set(dataset):
    scans_ts_ndx = {}

    for elem_ndx in dataset:
        ts, traversal = get_ts_traversal(dataset[elem_ndx]['query'])
        scans_ts_ndx[ts] = traversal

    return scans_ts_ndx


def create_lidar2img_ndx(lidar_timestamps, image_timestamps, k, threshold):
    print('Creating lidar2img index...')
    delta_l = []
    lidar2img_ndx = {}
    count_above_thresh = 0

    for lidar_ts in tqdm.tqdm(lidar_timestamps):
        nn_ndx = find_k_closest(image_timestamps, lidar_ts, k)
        nn_ts = image_timestamps[nn_ndx]
        nn_dist = np.abs(nn_ts - lidar_ts)
        #assert (nn_dist <= threshold).all(), 'nn_dist: {}'.format(nn_dist / 1000)
        # threshold is in miliseconds
        if (nn_dist > threshold*1000).sum() > 0:
            count_above_thresh += 1

        # Remember timestamps of closest images
        lidar2img_ndx[lidar_ts] = list(nn_ts)
        delta_l.extend(list(nn_dist))

    delta_l = np.array(delta_l, dtype=np.float32)
    s = 'Distance between the lidar scan and {} closest images (min/mean/max): {} / {} / {} [ms]'
    print(s.format(k, int(np.min(delta_l)/1000), int(np.mean(delta_l)/1000), int(np.max(delta_l)/1000)))
    print('{} scans without {} images within {} [ms] threshold'.format(count_above_thresh, k, int(threshold)))

    return lidar2img_ndx


# ***********************************

def find_cross_over(arr, low, high, x):
    if arr[high] <= x:  # x is greater than all
        return high

    if arr[low] > x:  # x is smaller than all
        return low

    # Find the middle point
    mid = (low + high) // 2  # low + (high - low)// 2

    # If x is same as middle element, then return mid
    if arr[mid] <= x and arr[mid + 1] > x:
        return mid

    # If x is greater than arr[mid], then either arr[mid + 1] is ceiling of x or ceiling lies in arr[mid+1...high]
    if arr[mid] < x:
        return find_cross_over(arr, mid + 1, high, x)

    return find_cross_over(arr, low, mid - 1, x)


def find_k_closest(arr, x, k):
    # This function returns indexes of k closest elements to x in arr[]
    n = len(arr)
    # Find the crossover point
    l = find_cross_over(arr, 0, n - 1, x)
    r = l + 1  # Right index to search
    ndx_l = []

    # If x is present in arr[], then reduce left index. Assumption: all elements in arr[] are distinct
    if arr[l] == x:
        l -= 1

    # Compare elements on left and right of crossover point to find the k closest elements
    while l >= 0 and r < n and len(ndx_l) < k:
        if x - arr[l] < arr[r] - x:
            ndx_l.append(l)
            l -= 1
        else:
            ndx_l.append(r)
            r += 1

    # If there are no more elements on right side, then print left elements
    while len(ndx_l) < k and l >= 0:
        ndx_l.append(l)
        l -= 1

    # If there are no more elements on left side, then print right elements
    while len(ndx_l) < k and r < n:
        ndx_l.append(r)
        r += 1

    return ndx_l

# *************************************


def convert_img(img_filepath, out_path, downsample=4):
    img_file = os.path.split(img_filepath)[1]
    out_filepath = os.path.join(out_path, img_file)

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    assert os.path.exists(out_path), 'Cannot create output folder: {}'.format(out_path)

    img = load_image(img_filepath)
    img = Image.fromarray(img)
    w, h = img.width, img.height
    img = img.resize((w // downsample, h // downsample))
    img.save(out_filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhance LiDAR dataset with corresponding RGB images.')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--oxford_root', type=str, required=True, help='Path to Oxford RobotCar dataset')
    parser.add_argument('--camera', type=str, default='stereo/centre', help='Camera')
    parser.add_argument('--out_path', type=str, required=True, help='Output path to save converted images')
    # Default downsampling factor 1280x960 images to 640x480
    parser.add_argument('--downsample', type=int, default=2, help='Image downsampling factor')

    args = parser.parse_args()
    print('Config path: {}'.format(args.config))
    print('Oxford RobotCar root folder: {}'.format(args.oxford_root))
    print('Camera: {}'.format(args.camera))
    print('Output path: {}'.format(args.out_path))
    print('Image downsampling factor: {}'.format(args.downsample))

    nn_threshold = 1000  # Nearest neighbour threshold in miliseconds
    k = 20               # Number of nearest neighbour images to find for each LiDAR scan
    ext = '.png'         # Image extension
    print('Number of nearest images for each scan (k): {}'.format(k))

    out_path = args.out_path
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    assert os.path.exists(out_path), 'Cannot create output directory: {}'.format(out_path)

    params = MinkLocParams(args.config, model_params_path=None)
    params.print()

    # Index LiDAR scans in the dataset
    lidar_ndx = index_scans_in_datasets(params)
    image_ndx = index_readings(args.oxford_root, args.camera, ext=ext)
    image_timestamps = [int(ts) for ts in list(image_ndx)]
    image_timestamps.sort()
    image_timestamps = np.array(image_timestamps, dtype=np.int64)

    lidar_timestamps = [int(ts) for ts in list(lidar_ndx)]
    lidar_timestamps.sort()
    lidar_timestamps = np.array(lidar_timestamps, dtype=np.int64)

    # Find k closest images for each lidar timestamp
    lidar2img_ndx_pickle = 'lidar2image_ndx.pickle'
    ndx_filepath = os.path.join(out_path, 'lidar2image_ndx.pickle')
    if os.path.exists(ndx_filepath):
        with open(ndx_filepath, 'rb') as f:
            lidar2img_ndx = pickle.load(f)
    else:
        lidar2img_ndx = create_lidar2img_ndx(lidar_timestamps, image_timestamps, k, nn_threshold)
        with open(ndx_filepath, 'wb') as f:
            pickle.dump(lidar2img_ndx, f)

    # Process NN images found (demosaic and rescale) and save in the designated folder
    print('Extracting and converting images...')

    args_l = []

    for lidar_ts in tqdm.tqdm(lidar2img_ndx):
        images_ndx = lidar2img_ndx[lidar_ts]
        for img_ndx in images_ndx:
            img_file_path = image_ndx[str(img_ndx)][0]
            assert os.path.exists(img_file_path), 'Cannot open image: {}'.format(img_file_path)
            traversal = image_ndx[str(img_ndx)][1]
            img_out_path = os.path.join(out_path, traversal)
            args_l.append((img_file_path, img_out_path, args.downsample))

    num_workers = 10
    print('Processing {} scenes using {} workers...'.format(len(args_l), num_workers))

    with Pool(num_workers) as p:
        p.starmap(convert_img, args_l)

    print('Finished')
