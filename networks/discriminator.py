"""
Defines the discriminator model, roughly based on the one from BigGAN.
"""

import torch

class Discriminator(torch.nn.Module):
    def __init__(self, dim):
        """
        Initializes the discriminative network.

        Note that this network is insensitive to the size of the input, 
        so some kind of multi-head attention/averaging has to be carried 
        before classification.

        Arguments :
            - dim (int) : dimension of the last vector, before mlp.
        """
        # define convnets

        # define averaging procedure

        # define final mlp

    def forward(img):
        """
        Forward pass.
        """