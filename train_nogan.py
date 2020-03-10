# No gan, here our loss is the reconstruction of the image conditioned on the
# provided part

import torch
import torchvision
import matplotlib.pyplot as plt

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder

from maskmaker import regular_transform, mask_transform
from generator import Generator
from discriminator import Encoder
from utils import n_params

# path to saved tensors, in batches of 512, loads quickly

true_path = op.join('data', 'tensors', 'original')
masked_path = op.join('data', 'tensors', 'masked')

# image folders, loads slowly, especially the masked one, use this for testing

Z_DIM = 64
N_EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3

# transforms for small-scale training

def regular_transform_small(img):
    img = torchvision.transforms.functional.resize(img, (100, 100))
    return regular_transform(img)

def mask_transform_small(img):
    img = torchvision.transforms.functional.resize(img, (100, 100))
    return mask_transform(img)

# dataloaders

imf_true = ImageFolder('data/images', regular_transform)
imf_masked = ImageFolder('data/images', mask_transform)

# train test split

# we assume true and masked have the same number of indices
indices = list(range(len(imf_true)))
split = len(imf_true) - 10
train_indices, test_indices = indices[:split], indices[split:]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

dl_true = DataLoader(
    imf_true,
    batch_size=BATCH_SIZE,
    shuffle=True)
dl_masked = DataLoader(
    imf_masked,
    batch_size=BATCH_SIZE,
    shuffle=True)

class Net(torch.nn.Module):
    def __init__(self, z_dim):
        super(Net, self).__init__()
        self.E = Encoder(z_dim)
        self.D = Generator(z_dim)

    def forward(m):
        z = self.E(m)
        return self.D(z, m)

net = Net(Z_DIM)

opt = torch.optim.Adam(net.parameters(), lr=LR)

loss_fn = torch.nn.MSELoss()

def train_reconstruction(N, net, opt, dl_true, dl_masked):
    for epoch in range(N):
        # batch is processed one by one for GPU mem reasons
        for batch_true, batch_masked in (dl_true, dl_masked):
            for t, m in zip(batch_true, batch_masked):
                # maybe encoding can be done in parallel
                # or encapsulate both models in one
                p = net(m)
                loss = loss_fn(p, t) / BATCH_SIZE
                loss.backward()
            opt.step()