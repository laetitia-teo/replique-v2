"""
This file defines the decoder, or generative network, of the GAN.

It is based on a masked version of the Spatial Broadcast Decoder.
"""

import torch

class Generator(torch.nn.Module):
    """
    Based on the Spatial Broadcast Decoder.
    """
    def __init__(self, zdim):
        self.zdim = zdim

        # define convnets

    def forward(z, image, mask):
        """
        Forward pass. Do we pass the random vector as input or do we generate
        it inside the function ?
        """
        # tile the z vector everywhere

        # add x, y position as features

        # add masked image as features