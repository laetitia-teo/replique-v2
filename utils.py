"""
Various utils.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_tensor_image(img, typ=int):
    """
    Some manips on a tensor image to plot it.
    """
    img = img[0]
    img = img.permute(1, 2, 0)
    try:
        img = img.detach().numpy().astype(typ)
    except TypeError:
        # cuda tensor
        img = img.cpu().detach().numpy().astype(typ)
    img = img[..., :3]
    plt.imshow(img)
    plt.show()

def n_params(model):
    return sum(p.numel() for p in model.parameters())

def make_yx(fmap, h, w, n):
    """
    We include the fmap in the inputs to be able to create the x and y tensors
    on the same device than the fmap.
    """
    y, x = np.meshgrid(
        np.arange(h, dtype=float),
        np.arange(w, dtype=float))
    y = fmap.new_tensor(y) # requires grad is false by default
    x = fmap.new_tensor(x)
    y = y / h
    x = x / w
    # add a dimension for batch
    y = y.unsqueeze(0) * fmap.new_ones((n, 1, 1))
    x = x.unsqueeze(0) * fmap.new_ones((n, 1, 1))
    y = y.unsqueeze(1)
    x = x.unsqueeze(1)
    return y, x