"""
This file defines the decoder, or generative network, of the GAN.

It is based on a masked version of the Spatial Broadcast Decoder.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

import utils as ut

from torchvision.datasets import ImageFolder
from maskmaker import mask_transform, regular_transform
from blocks import ResBlockG, GeneratorConv

class Generator(torch.nn.Module):
    """
    Based on the Spatial Broadcast Decoder.
    """
    def __init__(self, zdim):
        super(Generator, self).__init__()
        self.zdim = zdim

        # define convnets
        # this is an example, refine this
        channels = [
            (self.zdim, 64, 64),
            (64, 32, 32),
            (32, 32, 32),
            (32, 32, 32),
            (32, 32, 32),
            (32, 32, 32),
            (32, 32, 32),
            (32, 32, 32),
            (32, 32, 32),
            (32, 32, 32),
            (32, 32, 32),
            (32, 32, 3)]

        self.convnet = GeneratorConv(channels)

    def forward(self, z, img):
        """
        Arguments :
            - z : random vector for the generator
            - img : masked image concatenated with the mask
        """
        n, c, h, w = img.shape
        # tile the z vector everywhere
        f_map = z.new_ones((n, self.zdim, h, w))
        f_map *= z
        # x, y positions
        y, x = ut.make_yx(f_map, h, w, n)
        # process feature map with convnet
        f_map = self.convnet(f_map, img, x, y)
        f_map = (f_map + 1) * 0.5 # map to (0, 1) range
        # mix final image with the mask
        mask = img[:, -1, ...]
        f_map = f_map * mask + img[:, :-1, ...]
        return f_map

### Testing ###

from maskmaker import MaskMaker

def run():
    imf = ImageFolder('data/images', mask_transform)
    t = imf[2][0].unsqueeze(0)
    z = torch.rand((1, 100, 1, 1))
    gen = Generator(100)
    gen.cuda()
    t = t.cuda()
    z = z.cuda()
    img = gen(z, t)
    ut.plot_tensor_image(img, float)