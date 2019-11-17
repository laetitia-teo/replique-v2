"""
Module defining the different kinds of convolutional and devonvolutional blocks
used in our generator and discriminator.
Those blocks are strongly inspired by the ones used in the BigGAN paper.
"""
import torch

class ConvBlock(torch.nn.Module):
    """
    Convolutional block, with residual connections, used in the discriminator
    network.
    """
    def __init__(self):
        pass

    def forward(self, f_map):
        return NotImplemented