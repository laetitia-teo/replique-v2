"""
This class defines the dataset utilities for training.
"""
import time
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms.functional as TF

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder

max_size = (612, 612)

def custom_transform(img):
    img = TF.resize(img, max_size)
    # return TF.to_tensor(img)
    return img

# class ImageDataset(ImageFolder):
#     def __init__(self, root):
#         super(ImageDataset, self).__init__(root, transform)

imf = ImageFolder('data', custom_transform)
dl = DataLoader(imf, shuffle=True, batch_size=1500)
data_list = []

def time_one_pass(b_size):
    dl = DataLoader(imf, shuffle=True, batch_size=b_size)   
    t = time.time()
    for data, clss in dl:
        data.cuda()
        data.cpu()
    return time.time() - t

b_sizes = [32, 64, 128, 256, 512, 1024, 1500]
times = []
