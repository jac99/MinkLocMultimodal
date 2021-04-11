# Author: Jacek Komorowski
# Warsaw University of Technology

import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME

from datasets.oxford import OxfordDataset
from datasets.augmentation import TrainTransform, TrainSetTransform, TrainRGBTransform, ValRGBTransform
from datasets.samplers import BatchSampler
from misc.utils import MinkLocParams


def make_datasets(params: MinkLocParams, debug=False):
    # Create training and validation datasets
    datasets = {}
    train_transform = TrainTransform(params.aug_mode)
    train_set_transform = TrainSetTransform(params.aug_mode)

    if params.use_rgb:
        image_train_transform = TrainRGBTransform(params.aug_mode)
        image_val_transform = ValRGBTransform()
    else:
        image_train_transform = None
        image_val_transform = None

    if debug:
        max_elems = 1000
        val_max_elems = 1000

    else:
        # None means no limit
        max_elems = params.max_elems
        val_max_elems = params.val_max_elems

    datasets['train'] = OxfordDataset(params.dataset_folder, params.train_file, image_path=params.image_path,
                                      lidar2image_ndx=params.lidar2image_ndx, transform=train_transform,
                                      set_transform=train_set_transform, image_transform=image_train_transform,
                                      max_elems=max_elems, use_cloud=params.use_cloud)
    val_transform = None
    if params.val_file is not None:
        datasets['val'] = OxfordDataset(params.dataset_folder, params.val_file, image_path=params.image_path,
                                        lidar2image_ndx=params.lidar2image_ndx, transform=val_transform,
                                        set_transform=train_set_transform, image_transform=image_val_transform,
                                        max_elems=val_max_elems, use_cloud=params.use_cloud)
    return datasets


def make_collate_fn(dataset: OxfordDataset, mink_quantization_size=None):
    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        # Constructs a batch object
        labels = [e['ndx'] for e in data_list]

        # Compute positives and negatives mask
        # dataset.queries[label]['positives'] is bitarray
        positives_mask = [[dataset.queries[label]['positives'][e] for e in labels] for label in labels]
        negatives_mask = [[dataset.queries[label]['negatives'][e] for e in labels] for label in labels]

        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and
        # negatives_mask which are batch_size x batch_size boolean tensors
        result = {'positives_mask': positives_mask, 'negatives_mask': negatives_mask}

        if 'cloud' in data_list[0]:
            clouds = [e['cloud'] for e in data_list]

            clouds = torch.stack(clouds, dim=0)       # Produces (batch_size, n_points, 3) tensor
            if dataset.set_transform is not None:
                # Apply the same transformation on all dataset elements
                clouds = dataset.set_transform(clouds)

            coords = [ME.utils.sparse_quantize(coords=e, quantization_size=mink_quantization_size)
                      for e in clouds]
            coords = ME.utils.batched_coordinates(coords)
            # Assign a dummy feature equal to 1 to each point
            # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
            feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
            result['coords'] = coords
            result['features'] = feats

        if 'image' in data_list[0]:
            images = [e['image'] for e in data_list]
            result['images'] = torch.stack(images, dim=0)       # Produces (N, C, H, W) tensor

        return result

    return collate_fn


def make_dataloaders(params: MinkLocParams, debug=False):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets(params, debug=debug)

    dataloders = {}
    train_sampler = BatchSampler(datasets['train'], batch_size=params.batch_size,
                                 batch_size_limit=params.batch_size_limit,
                                 batch_expansion_rate=params.batch_expansion_rate)
    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    train_collate_fn = make_collate_fn(datasets['train'],  params.model_params.mink_quantization_size)
    dataloders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler, collate_fn=train_collate_fn,
                                     num_workers=params.num_workers, pin_memory=True)

    if 'val' in datasets:
        val_sampler = BatchSampler(datasets['val'], batch_size=params.val_batch_size)
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        val_collate_fn = make_collate_fn(datasets['val'], params.model_params.mink_quantization_size)
        dataloders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                       num_workers=params.num_workers, pin_memory=True)

    return dataloders
