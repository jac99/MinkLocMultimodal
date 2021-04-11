# Author: Jacek Komorowski
# Warsaw University of Technology

# Model processing LiDAR point clouds and RGB images

import torch
import torch.nn as nn
import torchvision.models as models
import MinkowskiEngine as ME

from models.minkloc import MinkLoc


class MinkLocMultimodal(torch.nn.Module):
    def __init__(self, cloud_fe, cloud_fe_size, image_fe, image_fe_size, output_dim: int,
                 fuse_method: str = 'concat', dropout_p: float = None, final_block: str = None):
        # cloud_fe: cloud feature extractor, returns tensor(batch_size, cloud_fe_size)
        # imaged_fe: image feature extractor, returns tensor(batch_size, image_fe_size)
        # output_dim: dimensionality of the fused global descriptor
        # dropout_p: whether to use Dropout after feature concatenation and before the fully connected block
        # add_fc_block: if True, a fully connected block is added after feature concatenation and optional Dropout block

        super().__init__()

        assert cloud_fe is not None or image_fe is not None

        self.cloud_fe = cloud_fe
        if cloud_fe is None:
            self.cloud_fe_size = 0
        else:
            self.cloud_fe_size = cloud_fe_size

        self.image_fe = image_fe
        if image_fe is None:
            self.image_fe_size = 0
        else:
            self.image_fe_size = image_fe_size

        self.output_dim = output_dim
        self.final_block = final_block
        self.dropout_p = dropout_p
        self.fuse_method = fuse_method

        if self.dropout_p is not None:
            self.dropout_layer = nn.Dropout(p=self.dropout_p)
        else:
            self.dropout_layer = None

        if fuse_method == 'concat':
            self.fused_dim = self.image_fe_size + self.cloud_fe_size
        elif fuse_method == 'add':
            assert self.image_fe_size == self.cloud_fe_size
            self.fused_dim = self.image_fe_size
        else:
            raise NotImplementedError('Unsupported fuse method: {}'.format(self.fuse_method))

        if self.final_block is None:
            self.final_net = None
        elif self.final_block == 'fc':
            self.final_net = nn.Linear(self.fused_dim, output_dim)
        elif self.final_block == 'mlp':
            temp_channels = self.output_dim
            self.final_net = nn.Sequential(nn.Linear(self.fused_dim, temp_channels, bias=False),
                                           nn.BatchNorm1d(temp_channels, affine=True),
                                           nn.ReLU(inplace=True), nn.Linear(temp_channels, output_dim))
        else:
            raise NotImplementedError('Unsupported final block: {}'.format(self.final_block))

    def forward(self, batch):
        y = {}
        if self.image_fe is not None:
            image_embedding = self.image_fe(batch)
            assert image_embedding.dim() == 2
            assert image_embedding.shape[1] == self.image_fe_size
            y['image_embedding'] = image_embedding

        if self.cloud_fe is not None:
            cloud_embedding = self.cloud_fe(batch)['embedding']
            assert cloud_embedding.dim() == 2
            assert cloud_embedding.shape[1] == self.cloud_fe_size
            y['cloud_embedding'] = cloud_embedding

        if self.cloud_fe is not None and self.image_fe is not None:
            assert cloud_embedding.shape[0] == image_embedding.shape[0]
            if self.fuse_method == 'concat':
                x = torch.cat([cloud_embedding, image_embedding], dim=1)
            elif self.fuse_method == 'add':
                assert cloud_embedding.shape == image_embedding.shape
                x = cloud_embedding + image_embedding
            else:
                raise NotImplementedError('Unsupported fuse method: {}'.format(self.fuse_method))
        elif self.cloud_fe is not None:
            x = cloud_embedding
        elif self.image_fe is not None:
            x = image_embedding

        if self.dropout_layer is not None:
            x = self.dropout_layer(x)

        if self.final_net is not None:
            x = self.final_net(x)

        assert x.shape[1] == self.output_dim, 'Output tensor has: {} channels. Expected: {}'.format(x.shape[1],
                                                                                                    self.output_dim)
        # x is (batch_size, output_dim) tensor
        y['embedding'] = x
        return y

    def print_info(self):
        print('Model class: MinkLocMultimodal')
        n_params = sum([param.nelement() for param in self.parameters()])
        print('Total parameters: {}'.format(n_params))
        if self.cloud_fe is not None:
            n_params = sum([param.nelement() for param in self.cloud_fe.parameters()])
            print('Cloud feature extractor parameters: {}'.format(n_params))

        if self.image_fe is not None:
            n_params = sum([param.nelement() for param in self.image_fe.parameters()])
            print('Image feature extractor parameters: {}'.format(n_params))

        print('Fuse method: {}'.format(self.fuse_method))
        if self.dropout_p is not None:
            print('Dropout p: {}'.format(self.dropout_p))

        print('Final block: {}'.format(self.final_block))
        if self.final_net is not None:
            n_params = sum([param.nelement() for param in self.final_net.parameters()])
            print('FC block parameters: {}'.format(n_params))

        print('Dimensionality of cloud features: {}'.format(self.cloud_fe_size))
        print('Dimensionality of image features: {}'.format(self.image_fe_size))
        print('Dimensionality of final descriptor: {}'.format(self.output_dim))


class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self, output_dim: int, add_fc_block: bool = False):
        super().__init__()
        self.output_dim = output_dim
        self.add_fc_block = add_fc_block

        model = models.resnet34(pretrained=True)
        # Last 2 blocks are AdaptiveAvgPool2d and Linear (get rid of them)
        self.resnet_fe = nn.ModuleList(list(model.children())[:-2])
        self.pool = GeM()
        if self.add_fc_block:
            self.fc = torch.nn.Linear(in_features=512, out_features=self.output_dim)

    def forward(self, batch):
        x = batch['images']

        # 0, 1, 2, 3 = first layers: Conv2d, BatchNorm, ReLu, MaxPool2d
        x = self.resnet_fe[0](x)
        x = self.resnet_fe[1](x)
        x = self.resnet_fe[2](x)
        x = self.resnet_fe[3](x)

        # sequential blocks, build from BasicBlock or Bottleneck blocks
        x = self.resnet_fe[4](x)
        x = self.resnet_fe[5](x)
        x = self.resnet_fe[6](x)
        x = self.resnet_fe[7](x)
        # x is (batch_size, 512, H=2, W=15) for 640x480 input image

        x = self.pool(x)
        # x is (batch_size, 512, 1, 1) tensor

        x = torch.flatten(x, 1)
        # x is (batch_size, 512) tensor

        if self.add_fc_block:
            x = self.fc(x)

        # (batch_size, feature_size)
        assert x.shape[1] == self.output_dim
        return x


class ResnetFPN(torch.nn.Module):
    def __init__(self, out_channels: int, lateral_dim: int, layers=[64, 64, 128, 256, 512], fh_num_bottom_up: int = 5,
                 fh_num_top_down: int = 2, add_fc_block: bool = False, pool_method='gem'):
        # Pooling types:  GeM, sum-pooled convolution (SPoC), maximum activations of convolutions (MAC)
        super().__init__()
        assert 0 < fh_num_bottom_up <= 5
        assert 0 <= fh_num_top_down < fh_num_bottom_up

        self.out_channels = out_channels
        self.lateral_dim = lateral_dim
        self.fh_num_bottom_up = fh_num_bottom_up
        self.fh_num_top_down = fh_num_top_down
        self.add_fc_block = add_fc_block
        self.layers = layers    # Number of channels in output from each ResNet block
        self.pool_method = pool_method.lower()
        model = models.resnet18(pretrained=True)
        # Last 2 blocks are AdaptiveAvgPool2d and Linear (get rid of them)
        self.resnet_fe = nn.ModuleList(list(model.children())[:3+self.fh_num_bottom_up])

        # Lateral connections and top-down pass for the feature extraction head
        self.fh_tconvs = nn.ModuleDict()    # Top-down transposed convolutions in feature head
        self.fh_conv1x1 = nn.ModuleDict()   # 1x1 convolutions in lateral connections to the feature head
        for i in range(self.fh_num_bottom_up - self.fh_num_top_down, self.fh_num_bottom_up):
            self.fh_conv1x1[str(i + 1)] = nn.Conv2d(in_channels=layers[i], out_channels=self.lateral_dim, kernel_size=1)
            self.fh_tconvs[str(i + 1)] = torch.nn.ConvTranspose2d(in_channels=self.lateral_dim,
                                                                  out_channels=self.lateral_dim,
                                                                  kernel_size=2, stride=2)

        # One more lateral connection
        temp = self.fh_num_bottom_up - self.fh_num_top_down
        self.fh_conv1x1[str(temp)] = nn.Conv2d(in_channels=layers[temp-1], out_channels=self.lateral_dim, kernel_size=1)

        # Pooling types:  GeM, sum-pooled convolution (SPoC), maximum activations of convolutions (MAC)
        if self.pool_method == 'gem':
            self.pool = GeM()
        elif self.pool_method == 'spoc':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.pool_method == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise NotImplementedError("Unknown pooling method: {}".format(self.pool_method))

        if self.add_fc_block:
            self.fc = torch.nn.Linear(in_features=self.lateral_dim, out_features=self.out_channels)

    def forward(self, batch):
        x = batch['images']
        feature_maps = {}

        # 0, 1, 2, 3 = first layers: Conv2d, BatchNorm, ReLu, MaxPool2d
        x = self.resnet_fe[0](x)
        x = self.resnet_fe[1](x)
        x = self.resnet_fe[2](x)
        x = self.resnet_fe[3](x)
        feature_maps["1"] = x

        # sequential blocks, build from BasicBlock or Bottleneck blocks
        for i in range(4, self.fh_num_bottom_up+3):
            x = self.resnet_fe[i](x)
            feature_maps[str(i-2)] = x

        assert len(feature_maps) == self.fh_num_bottom_up
        # x is (batch_size, 512, H=20, W=15) for 640x480 input image

        # FEATURE HEAD TOP-DOWN PASS
        xf = self.fh_conv1x1[str(self.fh_num_bottom_up)](feature_maps[str(self.fh_num_bottom_up)])
        for i in range(self.fh_num_bottom_up, self.fh_num_bottom_up - self.fh_num_top_down, -1):
            xf = self.fh_tconvs[str(i)](xf)        # Upsample using transposed convolution
            xf = xf + self.fh_conv1x1[str(i-1)](feature_maps[str(i - 1)])

        x = self.pool(xf)
        # x is (batch_size, 512, 1, 1) tensor

        x = torch.flatten(x, 1)
        # x is (batch_size, 512) tensor

        if self.add_fc_block:
            x = self.fc(x)

        # (batch_size, feature_size)
        assert x.shape[1] == self.out_channels
        return x


# GeM code adapted from: https://github.com/filipradenovic/cnnimageretrieval-pytorch

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
