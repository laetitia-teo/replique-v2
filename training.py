"""
This module defines the training loop ans all the utilities for training, such
as loading and saving models.
"""
import os.path as op
import torch
import torchvision
import matplotlib.pyplot as plt

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
z_dim = 16
LR = 10e-4 # tune this

# transforms for small-scale training

def regular_transform_small(img):
    img = torchvision.transforms.functional.resize(img, (100, 100))
    return regular_transform(img)

def mask_transform_small(img):
    img = torchvision.transforms.functional.resize(img, (100, 100))
    return mask_transform(img)

# dataloaders

imf_true = ImageFolder('data/images', regular_transform_small)
imf_masked = ImageFolder('data/images', mask_transform_small)

# train test split

# we assume true and masked have the same number of indices
indices = list(range(len(imf_true)))
split = len(imf_true) - 10
train_indices, test_indices = indices[:split], indices[split:]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

dl_true = DataLoader(
    imf_true,
    batch_size=B_SIZE,
    sampler=train_sampler)
dl_true_test = DataLoader(
    imf_true,
    batch_size=B_SIZE,
    sampler=test_sampler)
dl_masked = DataLoader(
    imf_masked,
    batch_size=B_SIZE,
    sampler=train_sampler)
dl_masked_test = DataLoader(
    imf_masked,
    batch_size=B_SIZE,
    sampler=test_sampler)

# metrics

def compute_accuracy(dl_true, dl_masked, D, G):
    D.cuda()
    G.cuda()
    with torch.no_grad():
        acc = 0
        for true_imgs, masked_imgs in tqdm(zip(dl_true, dl_masked)):
            bsize = len(true_imgs)
            a = 0
            true_imgs = true_imgs[0]
            masked_imgs = masked_imgs[0]
            for img, masked in tqdm(zip(true_imgs, masked_imgs)):
                img = img.unsqueeze(0)
                masked = masked.unsqueeze(0)
                if D(img) > 0.5:
                    a += 1 / 2
                z = torch.rand((1, z_dim, 1, 1))
                if D(G(z, masked)) <= 0.5:
                    a += 1 / 2
                print('a is %s' % a)
            acc += a / bsize
        return acc

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
        opt_D.zero_grad()

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
        opt_D.zero_grad()
    print('computing accuracies')
    print(compute_accuracy(dl_true_test, dl_masked_test, D, G))

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
            z = torch.rand((1, z_dim, 1, 1))
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

def train():
    # all on cpu
    criterion = torch.nn.BCELoss()
    D = Discriminator()
    G = Generator(z_dim)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR)
    opt_G = torch.optim.Adam(G.parameters(), lr=LR)
    fixed_zs = torch.rand((16, z_dim, 1, 1))
    fixed_img = imf_masked[2][0].unsqueeze(0)
    # batch counter
    i = 0
    for batch_true, batch_masked in zip(dl_true, dl_masked):
        batch_true = batch_true[0]
        batch_masked = batch_masked[0]
        bsize = len(batch_true)
        # optimize D
        D.zero_grad()
        for img_true, img_masked in zip(batch_true, batch_masked):
            img_true = img_true.unsqueeze(0)
            img_masked = img_masked.unsqueeze(0)
            # compute loss of discriminator on true img
            Dout = D(img_true)
            lossD_true = criterion(Dout, Dout.new_ones(Dout.shape)) / bsize
            lossD_true.backward()
            # compute loss on fake img
            with torch.no_grad():
                z = torch.rand((1, z_dim, 1, 1))
                img_fake = G(z, img_masked)
            Dout = D(img_fake)
            lossD_fake = criterion(Dout, Dout.new_zeros(Dout.shape)) / bsize
            lossD_fake.backward()
        opt_D.step()
        # optimize G
        G.zero_grad()
        for img_masked in batch_masked:
            img_masked = img_masked.unsqueeze(0)
            z = torch.rand((1, z_dim, 1, 1))
            fake_img = G(z, imf_masked)
            Dout = D(fake_img)
            lossG = criterion(Dout, Dout.new_ones(Dout.shape)) / bsize
            lossG.backward()
        opt_G.step()
        # end of each batch, save some images, and models
        with torch.no_grad():
            for j, z in enumerate(fixed_zs):
                img = G(z, fixed_img)
                img = torchvision.transforms.ToPILImage()(img).convert('RGB')
                img.save('saves/images/batch{0}_{1}.png'.format(i, j))
            # save models
            save_model(D, 'saves/models/D{0}.pt')
            save_model(G, 'saves/models/G{0}.pt')
        i += 1
# train(1)