"""
This module defines methods that take in an image and return the image
alongside a (connex, convex) mask that specifies which regions of the image
should be overlaid by the network. The mask is not purely boolean, any values
between 0 and 1 are accepted ; an intermediate value of alpha for a particular
pixel means the pixel of the resulting image, after processing by the generator
will be alpha * generated_image + (1 - alpha) * original_image.

This should allow for smooth transitions between the parts remaining from the
original image and the generated stuff.

The mask is generated (for now) by applying a succession of circular windows,
flat on top and equal to 1, with a Hann slope towards 0.

The windows are parametrized by their center c, their radius R and the length
of the downwards slope L. Several windows are used to create our mask, applied
by specifying the list of their parameters ; they can also be organized along
a curve to create a smooth connex region.
"""
import os.path as op
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF

from PIL import Image
from glob import glob

SIZE = (612, 612)

def window(dims=None, center=None, R=None, L=None):
    """
    Makes a window, given a numpy array, the center, the radius R and the 
    length of the slope L. All lengths are given in pixels.

    This function is not perfect, needs some form of antialiasing on 
    circle delimitation to get rid of artifacts.

    TODO : change this, its too slow. replace with clipping a function in 2d
    space, or something like that.
    """
    if dims is None:
        # convention : x dimension comes first
        w, h = 100, 100
    else:
        w, h = dims
    if center is None:
        center = np.array([50, 50])
    if R is None:
        R = 20
    if L is None:
        L = 10
    x, y = np.meshgrid(np.arange(h), np.arange(w))
    # center our coordinates
    x -= center[0]
    y -= center[1]
    # define our function
    f = lambda r: np.cos(np.pi * r / (2 * L))**2
    grid = np.zeros((w, h))
    hardmask = np.zeros((w, h))
    # add plateau
    print(x.shape)
    print(y.shape)
    print(R)
    grid += np.less_equal(x**2 + y**2, R**2)
    hardmask += np.less_equal(x**2 + y**2, R**2)
    for p in range(L):
        # define ring
        add = np.greater(x**2 + y**2, (R + p)**2)
        add *= np.less_equal(x**2 + y**2, (R + p + 1)**2)
        # hardmask += add.astype(float)
        # multiply by window value
        add = add.astype(float) * f(p)
        grid += add
    return grid, hardmask

def windowv2(dims, center, sigma, thresh):
    """
    New method for making a window.
    First we create a Gaussian centered on the provided center with the
    provided standard deviation. Then we clamp this gaussian at the provided
    threshold, and cast this to the (0-1) range.
    """
    w, h = dims
    x, y = np.meshgrid(np.arange(h), np.arange(w))
    x -= center[0]
    y -= center[1]
    grid = np.exp(- 2 * (x**2 + y**2)**0.5 / sigma)
    grid = np.clip(grid, 0, thresh)
    f = lambda x: (1 - np.cos(np.pi * x / thresh))/2
    grid = f(grid)
    return grid

def several_windows(dims, param_list):
    """
    Makes several windows from the parameters specified in the list.
    The dimensions of the grid are specified apart from the rest.
    """
    w, h = dims
    main_grid = np.zeros((w, h))
    main_hardmask = np.zeros((w, h))
    eps = 10e-8
    f = lambda x: 1 + np.tan(np.pi * x / (4 + eps))
    for params in param_list:
        grid = windowv2(*params)
        p = f(main_grid + grid)
        main_grid = (main_grid**p + grid**p)**(1/p)
        main_grid[main_grid > 1] = 1
    return main_grid

def sample_params(dims):
    # height and width are the same :
    size = dims[0]
    center = np.random.randint(0, size, 2)
    sigma = np.random.randint(size/6, size/2)
    thresh = np.random.random()
    return (dims, center, sigma, thresh)

def mask_transform(img):
    """
    Input is PIL image.
    Output is torch.Tensor
    """
    img = TF.resize(img, SIZE)
    params = []
    for _ in range(2):
        params.append(sample_params(img.size))
    # mask = np.zeros(SIZE)
    mask = several_windows(SIZE, params)
    t = TF.to_tensor(img)
    mask = torch.Tensor(mask).unsqueeze(0)
    t = t * (1 - mask)
    # mask = torch.ones(mask.shape)
    t = torch.cat((t, mask), 0)
    return t

def regular_transform(img):
    img = TF.resize(img, SIZE)
    return TF.to_tensor(img)
    return img

class Node():
    """
    The Nodes are situated between the different necks, and have a position in
    2d pixel (integer) space as well as in radius space (all assembled into the
    pos parameter).

    A Node keeps a reference to the following Necks. If those references are
    None, the Node is a head (or leaf).
    """
    def __init__(self, main, side, pos, R, L, n, inertia):
        self.main = main
        self.side = side
        self.pos = pos
        self.R = R
        self.L = L
        self.n = n

    def sample(self, N):
        """
        Recursively calls the sample method of the main and side Necks.
        """
        param_list = []
        if main is not None:
            param_list += self.main.sample(N)
        if side is not None:
            param_list += self.side.sample(N)
        return param_list

    def grow(self, mask):
        """
        Applied to a head, grows two additional necks, according to sampling 
        rules depending on the coverage of the image and the propagated neck
        number.
        """
        if self.main is None and self.side is None:
            dims = mask.shape
            # percentage of the grid covered by the mask
            p = np.sum(mask) / (dims[0] * dims[1]) 
            p = (2 * p)**2 # ?
            if p > 1:
                p = 1
            stop = np.random.binomial(1, p)
            if stop:
                return
            d = np.random.randint(0, 2)
            if not d:
                # left
                angle = (np.pi / 2) * np.random.beta(1, 3)

class Neck():
    """
    The class that really stores the information, such as the neck number, or
    the distribution of raduises along this neck. Also provides a method for 
    sampling centers along the neck. Keeps reference to the destination node.
    (the source is useless, since we only traverse the snake-tree in forward)
    
    This Neck class produces piecewise linear snake-trees, subclass this to
    create more convoluted bodies, based on higher-order splines for example. 
    """
    def __init__(self, src, dest, n):
        self.dest = dest
        self.src = src
        d = (self.src.pos - self.dest.pos)
        self.length = (d[0]**2 + d[1]**2)**.5
        self.n = n

    def sample(self, N):
        param_list = []
        for i in range(self.length / N):
            gamma = N / self.length
            center = gamma * self.src.pos + (1 - gamma) * self.dest.pos
            R = gamma * self.src.R + (1 - gamma) * self.dest.R
            L = gamma * self.src.L + (1 - gamma) * self.dest.L
            param_list.append((center, R, L))
        param_list += self.dest.sample(N)
        return param_list

class SnakeTree():
    """
    This class defines the snake-tree object, a geometrical object that lives
    on an image, and that can be used to produce a non-comvex complex mask.
    """
    def __init__(self, img):
        self.img = img

class MaskMaker():
    """
    The maskmaker class.
    """
    def __init__(self, img_paths):
        """
        Init.

        Arguments :

            - img_paths : list of paths to all the image folders.
        """
        self.img_paths = img_paths

    def load_images(self):
        paths = []
        for d in self.img_paths:
            paths += glob(op.join(d, '*.jpg')) # images have to be provided as jpg

    def sample_two(self):
        params = []
        for i in range(1):
            means = np.zeros(4)
            vec = (np.clip(np.random.normal(means), -1, 1) + 1) / 2
            vec /= 2
            vec += .5
            params.append(vec)
        return params

    def sample_params(self, dims):
        # height and width are the same :
        center = np.random.randint(0, dims[0], 2)
        sigma = np.random.randint(100, 300)
        thresh = np.random.random()
        return (dims, center, sigma, thresh)

    # def mask_transform()

    def make_one(self, path):
        # ugliest code :(
        img = Image.open(path)
        img = np.array(img)
        img = np.swapaxes(img, 0, 1)
        dims = img.shape[:-1]

        # diag = (h**2 + w**2)**.5
        # param_list = self.sample_two()
        # new_params = []
        # for vec in param_list:
        #     x = int(vec[0] * w)
        #     y = int(vec[1] * h)
        #     R = int(vec[2] * diag / 4)
        #     L = int(vec[3] * diag / 8)
        #     new_params.append(((x, y), R, L))
        mask, _ = several_windows((w, h), new_params)
        inverse_mask = 1 - mask
        masked = img * np.expand_dims(inverse_mask, -1)
        return masked, img, mask

    def mask_all(self, path):
        img = Image.open(path)
        img = np.array(img)
        img = np.swapaxes(img, 0, 1)
        w, h = img.shape[:-1]
        mask = np.ones((w, h))
        inverse_mask = 1 - mask
        masked = img * np.expand_dims(inverse_mask, -1)
        return masked, img, mask

m = MaskMaker(2)
# paramlist = [((25, 25), 45, 5), ((75, 10), 10, 50)]
# mg, hm = several_windows((100, 100), paramlist)
# plt.matshow(mg)
# plt.show()