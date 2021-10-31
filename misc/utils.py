# Author: Jacek Komorowski
# Warsaw University of Technology

import os
import configparser
import time
import pickle


class ModelParams:
    def __init__(self, model_params_path):
        assert os.path.exists(model_params_path), 'Cannot find model-specific configuration file: {}'.format(model_params_path)
        config = configparser.ConfigParser()
        config.read(model_params_path)
        params = config['MODEL']

        self.model_params_path = model_params_path
        self.model = params.get('model')
        self.mink_quantization_size = params.getfloat('mink_quantization_size', 0.01)

    def print(self):
        print('Model parameters:')
        param_dict = vars(self)
        for e in param_dict:
            print('{}: {}'.format(e, param_dict[e]))

        print('')


def get_datetime():
    return time.strftime("%Y%m%d_%H%M")


class MinkLocParams:
    """
    Params for training MinkLoc models on Oxford dataset
    """
    def __init__(self, params_path, model_params_path=None):
        """
        Configuration files
        :param path: General configuration file
        :param model_params: Model-specific configuration
        """

        assert os.path.exists(params_path), 'Cannot find configuration file: {}'.format(params_path)
        self.params_path = params_path
        self.model_params_path = model_params_path

        config = configparser.ConfigParser()

        config.read(self.params_path)
        params = config['DEFAULT']
        self.num_points = params.getint('num_points', 4096)
        self.dataset_folder = params.get('dataset_folder')
        self.use_cloud = params.getboolean('use_cloud', True)

        if 'image_path' in params:
            # Train with RGB
            # Evaluate on Oxford only (no images for InHouse datasets)
            self.use_rgb = True
            self.image_path = params.get('image_path')
            if 'lidar2image_ndx_path' not in params:
                self.lidar2image_ndx_path = os.path.join(self.image_path, 'lidar2image_ndx.pickle')
            else:
                self.lidar2image_ndx_path = params.get('lidar2image_ndx_path')

            self.eval_database_files = ['oxford_evaluation_database.pickle']
            self.eval_query_files = ['oxford_evaluation_query.pickle']
        else:
            # LiDAR only training and evaluation
            # Evaluate on Oxford and InHouse datasets
            self.use_rgb = False
            self.image_path = None
            self.lidar2image_ndx = None
            self.eval_database_files = ['oxford_evaluation_database.pickle', 'business_evaluation_database.pickle',
                                        'residential_evaluation_database.pickle', 'university_evaluation_database.pickle']

            self.eval_query_files = ['oxford_evaluation_query.pickle', 'business_evaluation_query.pickle',
                                     'residential_evaluation_query.pickle', 'university_evaluation_query.pickle']

        assert len(self.eval_database_files) == len(self.eval_query_files)

        params = config['TRAIN']
        self.num_workers = params.getint('num_workers', 0)
        self.batch_size = params.getint('batch_size', 128)
        # Validation batch size is fixed and does not grow
        self.val_batch_size = params.getint('val_batch_size', 64)

        # Set batch_expansion_th to turn on dynamic batch sizing
        # When number of non-zero triplets falls below batch_expansion_th, expand batch size
        self.batch_expansion_th = params.getfloat('batch_expansion_th', None)
        if self.batch_expansion_th is not None:
            assert 0. < self.batch_expansion_th < 1., 'batch_expansion_th must be between 0 and 1'
            self.batch_size_limit = params.getint('batch_size_limit', 256)
            # Batch size expansion rate
            self.batch_expansion_rate = params.getfloat('batch_expansion_rate', 1.5)
            assert self.batch_expansion_rate > 1., 'batch_expansion_rate must be greater than 1'
        else:
            self.batch_size_limit = self.batch_size
            self.batch_expansion_rate = None

        self.lr = params.getfloat('lr', 1e-3)
        # lr for image feature extraction
        self.image_lr = params.getfloat('image_lr', 1e-4)

        self.optimizer = params.get('optimizer', 'Adam')
        self.scheduler = params.get('scheduler', 'MultiStepLR')
        if self.scheduler is not None:
            if self.scheduler == 'CosineAnnealingLR':
                self.min_lr = params.getfloat('min_lr')
            elif self.scheduler == 'MultiStepLR':
                scheduler_milestones = params.get('scheduler_milestones')
                self.scheduler_milestones = [int(e) for e in scheduler_milestones.split(',')]
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(self.scheduler))

        self.epochs = params.getint('epochs', 20)
        self.weight_decay = params.getfloat('weight_decay', None)
        self.normalize_embeddings = params.getboolean('normalize_embeddings', True)    # Normalize embeddings during training and evaluation
        self.loss = params.get('loss')
        if self.loss == 'MultiBatchHardTripletMarginLoss':
            # Weights of different loss component
            weights = params.get('weights', '.3, .3, .3')
            self.weights = [float(e) for e in weights.split(',')]

        if 'Contrastive' in self.loss:
            self.pos_margin = params.getfloat('pos_margin', 0.2)
            self.neg_margin = params.getfloat('neg_margin', 0.65)
        elif 'Triplet' in self.loss:
            self.margin = params.getfloat('margin', 0.4)    # Margin used in loss function
        else:
            raise 'Unsupported loss function: {}'.format(self.loss)

        self.aug_mode = params.getint('aug_mode', 1)    # Augmentation mode (1 is default)

        self.train_file = params.get('train_file')
        self.val_file = params.get('val_file', None)

        # Read model parameters
        if self.model_params_path is not None:
            self.model_params = ModelParams(self.model_params_path)
        else:
            self.model_params = None

        self._check_params()

    def _check_params(self):
        assert os.path.exists(self.dataset_folder), 'Cannot access dataset: {}'.format(self.dataset_folder)

    def print(self):
        print('Parameters:')
        param_dict = vars(self)
        for e in param_dict:
            if e not in ['model_params']:
                print('{}: {}'.format(e, param_dict[e]))

        if self.model_params is not None:
            self.model_params.print()
        print('')

