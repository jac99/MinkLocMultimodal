# Author: Jacek Komorowski
# Warsaw University of Technology

from models.minkloc import MinkLoc
from models.minkloc_rgb import MinkLocRGB, ResnetFPN
from misc.utils import MinkLocParams


def model_factory(params: MinkLocParams):
    in_channels = 1

    # MinkLocMultimodal is our baseline MinkLoc++ model producing 256 dimensional descriptor where
    # each modality produces 128 dimensional descriptor
    # MinkLocRGB and MinkLoc3D are single-modality versions producing 256 dimensional descriptor
    if params.model_params.model == 'MinkLocMultimodal':
        cloud_fe_size = 128
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='ECABasicBlock', pooling_method='GeM')
        image_fe_size = 128
        image_fe = ResnetFPN(out_channels=image_fe_size, lateral_dim=image_fe_size,
                             fh_num_bottom_up=4, fh_num_top_down=0)
        model = MinkLocRGB(cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim=cloud_fe_size+image_fe_size)
    elif params.model_params.model == 'MinkLoc3D':
        cloud_fe_size = 256
        cloud_fe = MinkLoc(in_channels=1, feature_size=cloud_fe_size, output_dim=cloud_fe_size,
                           planes=[32, 64, 64], layers=[1, 1, 1], num_top_down=1,
                           conv0_kernel_size=5, block='ECABasicBlock', pooling_method='GeM')
        model = MinkLocRGB(cloud_fe, cloud_fe_size, None, 0, output_dim=cloud_fe_size,
                           dropout_p=None)
    else:
        raise NotImplementedError('Model not implemented: {}'.format(params.model_params.model))

    return model
