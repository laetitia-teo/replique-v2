"""
Various utils.
"""
import torch
import matplotlib.pyplot as plt


def plot_tensor_image(img, typ=int):
    """
    Some manips on a tensor image to plot it.
    """
    img = img[0]
    img = img.permute(1, 2, 0)
    img = img.detach().numpy().astype(typ)
    img = img[..., :3]
    plt.imshow(img)
    plt.show()