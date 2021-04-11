# Implementation of Efficient Channel Attention ECA block

import numpy as np
import torch.nn as nn

import MinkowskiEngine as ME

from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck


class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((np.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = ME.MinkowskiGlobalPooling()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y_sparse = self.avg_pool(x)

        # Apply 1D convolution along the channel dimension
        y = self.conv(y_sparse.F.unsqueeze(-1).transpose(-1, -2)).transpose(-1, -2).squeeze(-1)
        # y is (batch_size, channels) tensor

        # Multi-scale information fusion
        y = self.sigmoid(y)
        # y is (batch_size, channels) tensor

        y_sparse = ME.SparseTensor(y, coordinate_manager=y_sparse.coordinate_manager,
                                   coordinate_map_key=y_sparse.coordinate_map_key)
        # y must be features reduced to the origin
        return self.broadcast_mul(x, y_sparse)


class ECABasicBlock(BasicBlock):
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 dimension=3):
        super(ECABasicBlock, self).__init__(
            inplanes,
            planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            dimension=dimension)
        self.eca = ECALayer(planes, gamma=2, b=1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.eca(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
