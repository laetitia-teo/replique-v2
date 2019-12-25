"""
This module defines the training loop ans all the utilities for training, such
as loading and saving models.
"""
import os.path as op
import torch

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder

from maskmaker import regular_transform, mask_transform
from generator import Generator
from discriminator import Discriminator
from utils import n_params

# path to saved tensors, in batches of 512, loads quickly

true_path = op.join('data', 'tensors', 'original')
masked_path = op.join('data', 'tensors', 'masked')

# image folders, loads slowly, especially the masked one, use this for testing

B_SIZE = 4
z_dim = 100
LR = 10e-3 # tune this

# dataloaders

imf_true = ImageFolder('data/images', regular_transform)
imf_masked = ImageFolder('data/images', mask_transform)

# train test split

# we assume true and masked have the same number of indices
indices = list(range(len(imf_true)))
split = len(imf_true) - 1000
train_indices, test_indices = indices[:split], indices[split:]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

dl_true = DataLoader(
    imf_true,
    batch_size=B_SIZE,
    shuffle=True,
    sampler=train_sampler)
dl_true_test = DataLoader(
    imf_true,
    batch_size=B_SIZE,
    shuffle=True,
    sampler=test_sampler)
dl_masked = DataLoader(
    imf_masked,
    batch_size=B_SIZE,
    shuffle=True,
    sampler=train_sampler)
dl_masked_test = DataLoader(
    imf_masked,
    batch_size=B_SIZE,
    shuffle=True,
    sampler=test_sampler)

# metrics

def compute_accuracy(dl_true, dl_masked, D):
    with torch.no_grad():
        acc = 0
        for true_imgs, masked_imgs in zip(dl_true, dl_masked):

            for img, masked in zip(true_imgs, masked_imgs)

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

def trainD():
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

def trainDG():
    """
    We test training D on images generated by G, no gradients are computed
    for G.
    """
    D = Discriminator()
    opt_D = torch.optim.Adam(D.parameters(), lr=LR)
    G = Generator(z_dim)
    i = 0
    for true_imgs, masked_imgs in zip(dl_true, dl_masked):
        true_imgs = true_imgs[0]
        masked_imgs = masked_imgs[0]
        print('batch %s' % i)
        i += 1
        j = 0
        L = 0
        for true_img, masked_img in zip(true_imgs, masked_imgs):
            true_img = true_img.unsqueeze(0)
            masked_img = masked_img.unsqueeze(0)
            print('image %s' % j)
            j += 1
            print('computing fake image')
            with torch.no_grad():
                z = torch.rand((1, 100, 1, 1))
                fake_img = G(z, masked_img)
            print('done')
            print('loss1')
            loss = - torch.log(D(true_img)) / B_SIZE
            L += loss.item()
            print('loss1 backward')
            loss.backward()
            print('loss2')
            loss = - torch.log(1 - D(fake_img)) / B_SIZE
            L += loss.item()
            print('loss2 backward')
            loss.backward()
        print('batch loss %s' % L)
        print('optimizer step')
        opt_D.step()

def train_complete():
    """
    We test training D on images generated by G, no gradients are computed
    for G.
    """
    D = Discriminator()
    opt_D = torch.optim.Adam(D.parameters(), lr=LR)
    G = Generator(z_dim)
    i = 0
    for true_imgs, masked_imgs in zip(dl_true, dl_masked):
        true_imgs = true_imgs[0]
        masked_imgs = masked_imgs[0]
        print('batch %s' % i)
        i += 1
        j = 0
        for true_img, masked_img in zip(true_imgs, masked_imgs):
            true_img = true_img.unsqueeze(0)
            masked_img = masked_img.unsqueeze(0)
            print('image %s' % j)
            j += 1
            print('computing fake image')
            with torch.no_grad():
                z = torch.rand((1, 100, 1, 1))
                fake_img = G(z, masked_img)
            print('done')
            print('loss1')
            loss = - torch.log(D(true_img))
            print('loss1 backward')
            loss.backward()
            print('loss2')
            loss = - torch.log(1 - D(fake_img))
            print('loss2 backward')
            loss.backward()
        print('optimizer step')
        opt_D.step()

# train(1)