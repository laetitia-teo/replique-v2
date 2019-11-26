"""
This module defines the different residual blocks for composing our generative
and discriminative model.
"""
import torch
import torch.nn.functional as F

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
        self.drop_ch = in_ch - out_ch

        self.conv1 = torcn.nn.Conv2d(in_ch, h, 1)
        self.conv2 = torch.nn.Conv2d(h, h, 3)
        self.conv3 = torch.nn.Conv2d(h, h, 3)
        self.conv4 = torcn.nn.Conv2d(h, out_ch, 1)

        self.batchnorm1 = torch.nn.BatchNorm2d(in_ch)
        self.batchnorm2 = torch.nn.BatchNorm2d(h)
        self.batchnorm3 = torch.nn.BatchNorm2d(h)
        self.batchnorm4 = torch.nn.BatchNorm2d(h)

    def forward(self, fmap):
        # convolutional path
        out = self.batchnorm1(f_map)
        out = F.relu(out)

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
        skip = fmap[..., :self.drop_ch]

        out = out + skip

        return out

class ResBlockD(torch.nn.Module):
    """
    Residual block for the discriminator.
    """
    def __init__(self, in_ch, h, out_ch):
        super(ResBlockD, self).__init__()

        self.add_ch = out_ch - in_ch

        self.conv1 = torcn.nn.Conv2d(in_ch, h, 1)
        self.conv2 = torch.nn.Conv2d(h, h, 3)
        self.conv3 = torch.nn.Conv2d(h, h, 3)
        self.conv4 = torcn.nn.Conv2d(h, out, 1)

        self.conv5 = torch.nn.Conv2d(in_ch, self.add_ch, 1)

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

        # change again number of convolutions
        out = self.conv4(out)

        # skip connexion
        add = self.conv5(f_map)
        skip = torch.cat([f_map, add], -1)

        out = out + skip

        return out