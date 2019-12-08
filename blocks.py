"""
This module defines the different residual blocks for composing our generative
and discriminative model.
"""
import torch
import torch.nn.functional as F

from torch.nn import ModuleList

import utils as ut

### Residual Blocks ###

class ResBlockG(torch.nn.Module):
    """
    Base residual block for the generator.
    """
    def __init__(self, in_ch, h, out_ch):
        """
        Initializes the block.

        The block is mainly composed of two 3x3 convolutions, with 1x1
        convolutions at the beginning and end to change the numberts of
        channels as appropriate.

        h is the "hidden" internal number of channels, smaller than input or
        output channels in general, to have less parameters.
        """
        super(ResBlockG, self).__init__()
        # number of channels to drop in skip connexion
        self.ch_diff = out_ch - in_ch

        self.conv1 = torch.nn.Conv2d(in_ch, h, 1)
        self.conv2 = torch.nn.Conv2d(h, h, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(h, h, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(h, out_ch, 1)

        self.batchnorm1 = torch.nn.BatchNorm2d(in_ch)
        self.batchnorm2 = torch.nn.BatchNorm2d(h)
        self.batchnorm3 = torch.nn.BatchNorm2d(h)
        self.batchnorm4 = torch.nn.BatchNorm2d(h)

        if self.ch_diff > 0:
            self.cast = torch.nn.Conv2d(in_ch, self.ch_diff, 1)

    def forward(self, f_map):
        # convolutional path
        # out = self.batchnorm1(f_map)
        out = F.relu(f_map)
        # out = F.tanh(out)

        # change the number of channels
        out = self.conv1(out)
        out = self.batchnorm2(out)
        out = F.relu(out)

        # first regular convolution
        out = self.conv2(out)
        out = self.batchnorm3(out)
        out = F.relu(out)

        # second one
        out = self.conv3(out)
        out = self.batchnorm4(out)
        out = F.relu(out)

        # change, again, the number of channels
        out = self.conv4(out)

        # skip connexion
        if self.ch_diff <= 0:
            skip = f_map[:, :out.shape[1], ...]
        else:
            skip = torch.cat([f_map, self.cast(f_map)], 1)

        out = out + skip

        return out

class ResBlockD(torch.nn.Module):
    """
    Residual block for the discriminator.
    """
    def __init__(self, in_ch, h, out_ch, downsample=False):
        super(ResBlockD, self).__init__()

        self.downsample = downsample
        self.ch_diff = out_ch - in_ch

        self.conv1 = torch.nn.Conv2d(in_ch, h, 1)
        self.conv2 = torch.nn.Conv2d(h, h, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(h, h, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(h, out_ch, 1)

        if self.ch_diff > 0:
            self.conv5 = torch.nn.Conv2d(in_ch, self.ch_diff, 1)

        if self.downsample:
            self.avgpool = torch.nn.AvgPool2d(2)

    def forward(self, f_map):
        """
        Forward pass.
        """
        out = F.relu(f_map)

        # change number of channels
        out = self.conv1(out)
        out = F.relu(out)

        # regular convolutions
        out = self.conv2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = F.relu(out)

        # average pooling
        if self.downsample:
            out = self.avgpool(out)

        # change again number of convolutions
        out = self.conv4(out)

        # skip connexion
        if self.downsample:
            add = self.avgpool(f_map)
        else:
            add = f_map
        if self.ch_diff > 0:
            add2 = self.conv5(add)
            add = torch.cat([add, add2], 1)

        out = out + add

        return out

### Misc ###

def mlp_fn(layer_list):
    layers = []
    def mlp(f_in, f_out):
        f1 = f_in
        for f in layer_list:
            layers.append(torch.nn.Linear(f1, f))
            layers.append(torch.nn.ReLU())
            f1 = f
        layers.append(torch.nn.Linear(f1, f_out))
        return torch.nn.Sequential(*layers)
    return mlp

class AggBlock(torch.nn.Module):
    """
    Aggregation block.

    No learnable parameters, we simply use the last feature of the feature map
    as a weight to average all the stuff.
    """
    def __init__(self):
        super(AggBlock, self).__init__()

    def forward(self, f_map):
        # make sure all this works
        a_map = torch.sigmoid(f_map[:, -1, ...])
        denom = torch.sum(a_map, (2, 3))
        f_map = torch.sum(f_map[:, :-1, ...] * a_map, (2, 3))
        return f_map / denom

class AggBlockv2(torch.nn.Module):
    """
    Second version of the aggregation block.
    In this vesion we perform 1d conv on all vectors of the feature map to
    predict an attention map used as weights in the mean.

    We also use spatial (x, y) information as input to the aggregation
    convolution.
    """
    def __init__(self, in_ch):
        super(AggBlockv2, self).__init__()

        self.conv = torch.nn.Conv2d(in_ch + 2, 1, 1)

    def forward(self, f_map):
        n, c, h, w = f_map.shape
        y, x = ut.make_yx(h, w, n)
        f_map2 = torch.cat((f_map, x, y), 1)
        a_map = torch.sigmoid(self.conv(f_map2))
        denom = torch.sum(a_map, (2, 3))
        f_map = torch.sum(f_map * a_map, (2, 3))
        return f_map / denom

### Full Nets ###

class GeneratorConv(torch.nn.Module):
    """
    This class defines the conv net used in the generator.
    The net is not a simple sequential stacking of the residual blocks : in 
    addition to this we also need to feed the image (and mask ?) information
    at the beginning of each residual block (+4 dimensions).
    """
    def __init__(self, feature_list):
        """
        Takes as input the list of tuples defining the inputs to the residual
        block constructor.
        """
        super(GeneratorConv, self).__init__()
        # for i, (in_ch, h, out_ch) in enumerate(feature_list):
        #     # if i == 0:
        #     #     # TODO : change this
        #     #     in_ch += 2
        #     self.__dict__['block' + str(i)] = ResBlockG(in_ch + 6, h, out_ch)
        # use ModuleList
        self.mlist = ModuleList(
            [ResBlockG(in_ch + 6, h, out_ch) \
                for in_ch, h, out_ch in feature_list])
        # self.N = len(feature_list)

    def forward(self, inpt, img, x, y):
        """
        X and Y info are already in the input.
        """
        out = inpt
        # for i in range(self.N):
        #     # we concatenate, in the channel dim, image and mask info
        #     # print('block %s' % i)
        #     # ut.plot_tensor_image(out, float)
        #     out = self.__dict__['block' + str(i)](
        #         torch.cat((out, img, x, y), 1))
        for i, block in enumerate(self.mlist):
            print(i)
            out = block(torch.cat((out, img, x, y), 1))
        out = torch.tanh(out)
        return out