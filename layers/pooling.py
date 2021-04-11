# Pooling methods code based on: https://github.com/filipradenovic/cnnimageretrieval-pytorch
# Global covariance pooling methods implementation taken from:
# https://github.com/jiangtaoxie/fast-MPN-COV
# and ported to MinkowskiEngine by Jacek Komorowski

import torch
import torch.nn as nn
import MinkowskiEngine as ME


class PoolingWrapper(nn.Module):
    def __init__(self, pool_method, in_dim, output_dim):
        super().__init__()

        self.pool_method = pool_method
        self.in_dim = in_dim
        self.output_dim = output_dim
        # Requires conversion of Minkowski sparse tensor to a batch
        self.convert_to_batch = False

        if pool_method == 'MAC':
            # Global max pooling
            assert in_dim == output_dim
            self.pooling = MAC(input_dim=in_dim)
        elif pool_method == 'SPoC':
            # Global average pooling
            assert in_dim == output_dim
            self.pooling = SPoC(input_dim=in_dim)
        elif pool_method == 'GeM':
            # Generalized mean pooling
            assert in_dim == output_dim
            self.pooling = GeM(input_dim=in_dim)
        else:
            raise NotImplementedError('Unknown pooling method: {}'.format(pool_method))

    def forward(self, x: ME.SparseTensor):
        if self.convert_to_batch:
            x = make_feature_batch(x)

        return self.pooling(x)


class MAC(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = self.input_dim
        self.f = ME.MinkowskiGlobalMaxPooling()

    def forward(self, x: ME.SparseTensor):
        x = self.f(x)
        return x.F      # Return (batch_size, n_features) tensor


class SPoC(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = self.input_dim
        self.f = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: ME.SparseTensor):
        x = self.f(x)
        return x.F      # Return (batch_size, n_features) tensor


class GeM(nn.Module):
    def __init__(self, input_dim, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.input_dim = input_dim
        # Same output number of channels as input number of channels
        self.output_dim = self.input_dim
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.f = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: ME.SparseTensor):
        # This implicitly applies ReLU on x (clamps negative values)
        temp = ME.SparseTensor(x.F.clamp(min=self.eps).pow(self.p), coordinates=x.C)
        temp = self.f(temp)             # Apply ME.MinkowskiGlobalAvgPooling
        return temp.F.pow(1./self.p)    # Return (batch_size, n_features) tensor


def make_feature_batch(x: ME.SparseTensor):
    # Covert sparse features into a batch of size (batch_size, N, channels) padded with zeros to ensure the same
    # number of feature in each element
    features = x.decomposed_features

    # features is a list of (n_features, channels) tensors with variable number of points
    batch_size = len(features)
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    # features is (batch_size, n_features, n_channels) tensor padded with zeros
    features = features.permute(0, 2, 1).contiguous()
    # features is (batch_size, n_channels, n_features) tensor padded with zeros
    assert features.ndim == 3
    return features
