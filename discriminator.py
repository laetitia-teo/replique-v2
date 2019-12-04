"""
Defines the discriminator model, roughly based on the one from BigGAN.
"""

import torch

from blocks import ResBlockD, AggBlock, mlp_fn

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
        # this is random, tune it please
        channels = [
            (3, 8, 16),
            (16, 32, 32),
            (32, 64, 64),
            (64, 128, 128),
            (128, 128, 129)] # last channel is for averaging weights

        # # tune this
        # mlp = mlp_fn([128, 64])(128, 1)

        block_list = []
        for in_ch, h, out_ch in channels:
            block_list.append(ResBlockD(in_ch, h. out_ch))
        block_list.append(torch.nn.ReLU())
        block_list.append(AggBlock())
        block_list.append(torch.nn.Linear(128, 64))
        block_list.appemd(torch.nn.ReLU())
        block_list.append(torch.nn.Linear(64, 1))

        self.convnet = torch.nn.Sequential(*block_list)

    def forward(self, img):
        """
        Forward pass.
        """
        return self.convnet(img)