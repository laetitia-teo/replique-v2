"""
This module defines the training loop ans all the utilities for training, such
as loading and saving models.
"""
import os.path as op
import torch

from generator import Generator
from discriminator import Discriminator

true_path = op.join('data', 'tensors', 'original')
masked_path = op.join('data', 'tensors', 'masked')

z_dim = 100

G = Generator(z_dim)
D = Discriminator()