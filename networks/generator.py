"""
This file defines the decoder, or generative network, of the GAN.

It is based on a masked version of the Spatial Broadcast Decoder.
"""
import numpy as np
import torch



class Generator(torch.nn.Module):
    """
    Based on the Spatial Broadcast Decoder.
    """
    def __init__(self, zdim):
        self.zdim = zdim

        # define convnets

    def forward(z, img, mask):
        """
        Forward pass. Do we pass the random vector as input or do we generate
        it inside the function ? 
        Do we concatenate the masked image at each step, to keep the info ?

        Arguments :
            - z : random vector for the generator
            - img : masked image
            - mask : do we need this info in the feature map ?
        """
        n, h, w, c = img.shape
        # tile the z vector everywhere
        f_map = torch.ones((n, h, w, self.zdim))
        f_map *= z
        # x, y positions
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
        fill = torch.ones((n, h, w, 1))
        # we may need to add a dimension for batch here
        y /= h * fill
        x /= w * fill
        f_map = torch.cat((f_map, y, x, img, mask), -1) # check this
        # process feature map with transpose convolutions 
        ...
        # mix final image with the mask
        f_map = f_map * mask + img
        return f_map
