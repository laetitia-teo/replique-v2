"""
This module defines the training loop ans all the utilities for training, such
as loading and saving models.
"""
import os.path as op
import torch

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from maskmaker import regular_transform, mask_transform
from generator import Generator
from discriminator import Discriminator

# path to saved tensors, in batches of 512, loads quickly

true_path = op.join('data', 'tensors', 'original')
masked_path = op.join('data', 'tensors', 'masked')

# image folders, loads slowly, especially the masked one, use this for testing

B_SIZE = 128
z_dim = 100
LR = 10e-3 # tune this

imf_true = ImageFolder('data/images', regular_transform)
imf_masked = ImageFolder('data/images', masked_transform)
dl_true = DataLoader(imf_true, batch_size=B_SIZE, shuffle=True)
dl_masked = DataLoader(imf_masked, batch_size=B_SIZE, shuffle=True)

# models

G = Generator(z_dim)
D = Discriminator()

# training functions

def save_model(m, path):
    torch.save(m.state_dict(), path)

def load_model(m, path):
    m.load_state_dict(torch.load(path))

def train(n, models=None):
    """
    Training loop.

    n is number of epochs.
    """
    # sample a number of z vectors for making timelines.
    ref_z = torch.rand(10, z_dim)
    if models is None:
        G = Generator(z_dim)
        D = Discriminator()
    else:
        G, D = models
    opt_G = torch.optim.Adam(G.parameters(), lr=LR)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR)
    for step in range(n):
        for imgs, m_imgs in zip(dl_true, dl_masked):
            loss = 0
            for img, m_img in zip(imgs, m_imgs):
                # we can't fit more than 1 image in GPU RAM :'(
                # fix this, because discriminator does not output probas
                loss = - torch.log(D(img)) + torch.log(D(G(m_imgs)))
                loss /= B_SIZE
                loss.backward()
            opt_G.step()
            opt_D.step()
            torch.zero_grad() # was this the syntax ?
            # let's save our models after each batch to see what's up