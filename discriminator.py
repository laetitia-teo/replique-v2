"""
Defines the discriminator model, roughly based on the one from BigGAN.
"""

import torch

from blocks import ResBlockD, AggBlockv2, Cut, mlp_fn

class Discriminator(torch.nn.Module):
    def __init__(self):
        """
        Initializes the discriminative network.

        Note that this network is insensitive to the size of the input, 
        so some kind of multi-head attention/averaging has to be carried 
        before classification.

        Arguments :
            - dim (int) : dimension of the last vector, before mlp.
        """
        super(Discriminator, self).__init__()
        self.GPU = False
        # define convnets
        # this is random, tune it please
        channels = [
            (3, 8, 16),
            (16, 32, 32),
            (32, 64, 64),
            (64, 128, 128),
            (128, 128, 128)]

        downsampled = [3, 4]

        # # tune this
        # mlp = mlp_fn([128, 64])(128, 1)

        block_list = []
        for i, (in_ch, h, out_ch) in enumerate(channels):
            if i in downsampled:
                downsample = True
            else:
                downsample = False
            block_list.append(ResBlockD(in_ch, h, out_ch, downsample))
        block_list.append(torch.nn.ReLU())
        block_list.append(AggBlockv2(out_ch))
        block_list.append(torch.nn.Linear(out_ch, 64))
        block_list.append(torch.nn.ReLU())
        block_list.append(torch.nn.Linear(64, 1))

        self.convnet = torch.nn.Sequential(*block_list)

    def forward(self, img):
        """
        Forward pass.

        Returns probabilities.
        """
        return torch.sigmoid(self.convnet(img))

class DiscriminatorSimple(torch.nn.Module):
    """
    Simpler version for the discriminator, one that is sure to work (?)
    Similar in spirit to the above, but more downsampling of the feature map,
    and a flattening layer operation followed by an mlp at the end, as is
    classical.

    Careful, with this guy we lose the invariance to size of input.
    For now we calibrate this to 
    We'll use this for test purposes.
    """
    def __init__(self):
        super(DiscriminatorSimple, self).__init__()
        self.GPU = False
        # define convnets
        # this is random, tune it please
        channels = [
            (3, 8, 16),
            (16, 32, 32),
            (32, 64, 64),
            (64, 64, 64),
            (64, 128, 128),
            (128, 128, 128),
            (128, 128, 128)]

        downsampled = [0, 1, 2, 3, 4, 5]

        # # tune this
        # mlp = mlp_fn([128, 64])(128, 1)

        block_list = []
        for i, (in_ch, h, out_ch) in enumerate(channels):
            if i in downsampled:
                downsample = True
            else:
                downsample = False
            block_list.append(ResBlockD(in_ch, h, out_ch, downsample))
            if i in [1, 4]:
                block_list.append(Cut((1, 1)))
        block_list.append(torch.nn.ReLU())
        block_list.append(torch.nn.Flatten())
        block_list.append(torch.nn.Linear(128 * 81, 64))
        block_list.append(torch.nn.ReLU())
        block_list.append(torch.nn.Linear(64, 64))
        block_list.append(torch.nn.ReLU())
        block_list.append(torch.nn.Linear(64, 1))

        self.convnet = torch.nn.Sequential(*block_list)

    def forward(self, img):
        """
        Forward pass.

        Returns probabilities.
        """
        return torch.sigmoid(self.convnet(img))

class Discriminatorv2(torch.nn.Module):
    """
    Simpler version for the discriminator, one that is sure to work (?)
    Similar in spirit to the above, but more downsampling of the feature map,
    and a flattening layer operation followed by an mlp at the end, as is
    classical.

    Careful, with this guy we lose the invariance to size of input.
    For now we calibrate this to 
    We'll use this for test purposes.
    """
    def __init__(self):
        super(Discriminatorv2, self).__init__()
        self.GPU = False
        # define convnets
        # this is random, tune it please
        channels = [
            (3, 8, 16),
            (16, 32, 32),
            (32, 64, 64),
            (64, 64, 64),
            (64, 128, 128),
            (128, 128, 128),
            (128, 128, 128)]

        downsampled = [0, 1, 2, 3, 4, 5]

        # # tune this
        # mlp = mlp_fn([128, 64])(128, 1)

        block_list = []
        for i, (in_ch, h, out_ch) in enumerate(channels):
            if i in downsampled:
                downsample = True
            else:
                downsample = False
            block_list.append(ResBlockD(in_ch, h, out_ch, downsample))
            if i in [1, 4]:
                block_list.append(Cut((1, 1)))
        block_list.append(torch.nn.ReLU())
        block_list.append(AggBlockv2(out_ch))
        block_list.append(torch.nn.Linear(64, 1))

        self.convnet = torch.nn.Sequential(*block_list)

    def forward(self, img):
        """
        Forward pass.

        Returns probabilities.
        """
        return torch.sigmoid(self.convnet(img))

class Encoder(torch.nn.Module):
    """
    Based on Discriminatorv2.
    """
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.GPU = False
        # define convnets
        # this is random, tune it please
        channels = [
            (3, 8, 16),
            (16, 32, 32),
            (32, 64, 64),
            (64, 64, 64),
            (64, 128, 128),
            (128, 128, 128),
            (128, 128, 128)]

        downsampled = [0, 1, 2, 3, 4, 5]

        # # tune this
        # mlp = mlp_fn([128, 64])(128, 1)

        block_list = []
        for i, (in_ch, h, out_ch) in enumerate(channels):
            if i in downsampled:
                downsample = True
            else:
                downsample = False
            block_list.append(ResBlockD(in_ch, h, out_ch, downsample))
            if i in [1, 4]:
                block_list.append(Cut((1, 1)))
        block_list.append(torch.nn.ReLU())
        block_list.append(AggBlockv2(out_ch))
        block_list.append(torch.nn.Linear(128, z_dim))

        self.convnet = torch.nn.Sequential(*block_list)

    def forward(self, img):
        """
        Forward pass.
        """
        return self.convnet(img)

if __name__ == '__main__':
    # test :
    from torchvision.datasets import ImageFolder
    from maskmaker import *

    imf = ImageFolder('data/images', regular_transform)
    img = imf[2][0].unsqueeze(0)
    D = Discriminator()
    res = D(img)

    from generator import Generator
    imf = ImageFolder('data/images', mask_transform)
    t = imf[2][0].unsqueeze(0)
    z = torch.rand((1, 100, 1, 1))
    G = Generator(100)
    G.cuda()
    t = t.cuda()
    z = z.cuda()
    img = G(z, t).cpu()
    res = D(img)