"""
This file defines the decoder, or generative network, of the GAN.

It is based on a masked version of the Spatial Broadcast Decoder.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

import utils as ut

from torchvision.datasets import ImageFolder
from maskmaker import mask_transform
from blocks import ResBlockG, GeneratorConv

class Generator(torch.nn.Module):
    """
    Based on the Spatial Broadcast Decoder.
    """
    def __init__(self, zdim):
        super(Generator, self).__init__()
        self.zdim = zdim

        # define convnets
        # this is an example, refine this, too many channels I think
        channels = [
            (self.zdim, 64, 64),
            (64, 64, 64),
            (64, 64, 64),
            (64, 64, 64),
            (64, 64, 64),
            (64, 64, 64),
            (64, 64, 64),
            (64, 64, 64),
            (64, 64, 64),
            (64, 32, 64),
            (64, 32, 32),
            (32, 32, 3)]

        self.convnet = GeneratorConv(channels)

    def forward(self, z, img):
        """
        Forward pass. Do we pass the random vector as input or do we generate
        it inside the function ? 
        Do we concatenate the masked image at each step, to keep the info ?

        there may be mpre intelligent ways to do the tiling, for instance we
        could sample n random vectors and tile the f_map with a weigthed mean
        of those vectors, where the weights would be computed based on the
        spatial position, in a transformer-fasion.

        Arguments :
            - z : random vector for the generator
            - img : masked image concatenated with the mask
        """
        n, c, h, w = img.shape
        # tile the z vector everywhere
        f_map = torch.ones((n, self.zdim, h, w))
        f_map *= z
        # x, y positions
        y, x = torch.meshgrid(
            torch.arange(h, dtype=torch.float),
            torch.arange(w, dtype=torch.float))
        fill = torch.ones((n, 1, h, w))
        y = y / h
        x = x / w
        # add a dimension for batch
        y = y.unsqueeze(0) * torch.ones((n, 1, 1))
        x = x.unsqueeze(0) * torch.ones((n, 1, 1))
        y = y.unsqueeze(1)
        x = x.unsqueeze(1)
        print(f_map.shape)
        # f_map = torch.cat((f_map), 1) # should work
        print(f_map.shape)
        print(img.shape)
        # process feature map with convnet
        f_map = self.convnet(f_map, img, x, y)
        f_map = (f_map + 1) * 127.5
        # mix final image with the mask
        mask = img[:, -1, ...]
        f_map = f_map * mask + img[:, :-1, ...] * 255
        return f_map

### Testing ###

from maskmaker import MaskMaker

# m = MaskMaker(2)
# mi, i, mask = m.mask_all('images/tests/image.jpg')
# mi = torch.tensor(np.expand_dims(mi, 0), dtype=torch.float)
# i = torch.tensor(np.expand_dims(i, 0), dtype=torch.float)
# mask = torch.tensor(np.expand_dims(mask, 0), dtype=torch.float)
# mask = mask.unsqueeze(0)
# mi = mi.permute(0, 3, 1, 2)
# i = i.permute(0, 3, 1, 2)
# mask = mask.permute(0, 1, 2, 3)
# gen = Generator(100)
# z = torch.rand((1, 100, 1, 1)) / 2
# # z = i.float() / 255
# img = gen(z, mi, mask)
# print('final')
# ut.plot_tensor_image(img)
def run():
    imf = ImageFolder('data/images', mask_transform)
    t = imf[2][0].unsqueeze(0)
    z = torch.rand((1, 100, 1, 1))
    gen = Generator(100)
    gen.cuda()
    t.cuda()
    z.cuda()
    img = gen(z, t)
    ut.plot_tensor_image(img)