import cv2
import torch
import pickle
from torch.utils import data
import numpy as np
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

class ImageDataset(data.Dataset):
    """
    Dataset class for handwritten lines.

    Parameters
    ----------
    pickle_file : str
        Path to a dataset pickle file.
    meta : bool
        If True, meta data about files is provided.
    num_imgs : int, optional (default=-1)
        Choose only `num_imgs` imgs for processing. If set to -1, uses all available images.
    rescale_mode : bool
        Enables rescaling.
    padding_mode : bool
        Enables data padding
    rescale_size : tuple of ints
        Size of image to rescale to.
    padding_size : tuple of ints
        Size of padding.

    """

    def __init__(self, pickle_file, meta=False, num_imgs=-1, rescale_mode=True, padding_mode=False, 
                 rescale_size=(442, 64), padding_size=(1024,128), aug_prob = 0.0):
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)
        self.reference_dataset = dataset
        self.num_imgs = num_imgs
        self.rescale_mode = rescale_mode
        self.padding_mode = padding_mode
        self.rescale_size = rescale_size
        self.padding_size = padding_size
        self.aug_prob = aug_prob
        if (self.num_imgs > 0) and (self.num_imgs < len(self.reference_dataset)):
            inds = np.random.choice(np.arange(len(self.reference_dataset)),
                                    self.num_imgs, replace=False)
            self.reference_dataset = [self.reference_dataset[i] for i in inds]
        self.meta = meta

    def __getitem__(self, idx):
        data = self.reference_dataset[idx]
        img = cv2.cvtColor(cv2.imread(data['img_path']), cv2.COLOR_BGR2RGB) / 255.
        if random.random()<self.aug_prob:
            img = elastic_transform(img, img.shape[1] * 0.5, img.shape[1] * 0.025, img.shape[0] * 0.01, random_state=None)
        if self.rescale_mode:
            img = cv2.resize(img, self.rescale_size)
        elif self.padding_mode:
            img_shape = img.shape
            koef_1 = img_shape[1]/self.padding_size[0]
            koef_2 = img_shape[0]/self.padding_size[1]
            use_koef = max(koef_1, koef_2)
            if use_koef > 1.:
                img = cv2.resize(img, (math.floor(img_shape[1]/use_koef), math.floor(img_shape[0]/use_koef)))
                img_shape = img.shape
            padded_image = np.ones((self.padding_size[1], self.padding_size[0], img_shape[2]))
            padded_image[0:img_shape[0], 0:img_shape[1], :] = img
            img = padded_image
            
        img = torch.from_numpy(img).permute(2,0,1).float()
        text = data['description']
        width = data['width']
        height = data['height']

        if self.meta:
            return img, text, data, width, height
        else:
            return img, text

    def __len__(self):
        return len(self.reference_dataset)
    
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REPLICATE)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)