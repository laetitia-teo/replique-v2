"""
This module defines the training loop ans all the utilities for training, such
as loading and saving models.
"""
import os.path as op
import torch

from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from maskmaker import regular_transform, mask_transform
from generator import Generator
from discriminator import Discriminator
from utils import nparams

# path to saved tensors, in batches of 512, loads quickly

true_path = op.join('data', 'tensors', 'original')
masked_path = op.join('data', 'tensors', 'masked')

# image folders, loads slowly, especially the masked one, use this for testing

B_SIZE = 4
z_dim = 100
LR = 10e-3 # tune this

imf_true = ImageFolder('data/images', regular_transform)
imf_masked = ImageFolder('data/images', mask_transform)
dl_true = DataLoader(imf_true, batch_size=B_SIZE, shuffle=True)
dl_masked = DataLoader(imf_masked, batch_size=B_SIZE, shuffle=True)

# models

# G = Generator(z_dim)
# D = Discriminator()

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
        for imgs, m_imgs in tqdm(zip(dl_true, dl_masked)):
            imgs = imgs[0]
            m_imgs = m_imgs[0]
            loss = 0
            for true, masked in zip(imgs, m_imgs):
                # we go through batch one image at a time
                # we can't fit more than 1 image in GPU RAM :'(
                # loss = torch.log(D(img)) + torch.log(D(G(m_imgs)))
                # true_img, _ = true
                # m_img, _ = masked
                print('one step')
                true = true.unsqueeze(0)
                masked = masked.unsqueeze(0) 
                with torch.no_grad():
                    # z = torch.rand((1, 100, 1, 1))
                    # fake = G(z, masked)
                    fake = true
                loss = torch.log(D(true)) + torch.log(1 - D(fake))
                # loss /= B_SIZE
                loss.backward()
                # opt_G.step()
                opt_D.step()
                opt_D.zero_grad()
            print('Epoch %s, batch %s, loss %s')
            # let's save our models after each batch to see what's up

def train_test():
    """
    Second, test version of the training loop, only trains the discriminator.
    """
    D = Discriminator()
    opt_D = torch.optim.Adam(D.parameters(), lr=LR)
    G = Generator(z_dim)
    i = 0
    for imgs, _ in dl_true:
        print('batch %s' % i)
        i += 1
        for j, img in enumerate(imgs):
            print('image %s' % j)
            img = img.unsqueeze(0)
            print('computing losses')
            loss1 = - torch.log(D(img)) / B_SIZE
            loss2 = - torch.log(1 - D(img)) / B_SIZE
            print('backward, destroying graphs')
            loss1.backward()
            loss2.backward()
        print('applying optimizer')
        opt_D.step()


# train(1)