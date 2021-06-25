import cv2
import torch
import pickle
from torch.utils import data
import numpy as np

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
                 rescale_size=(442, 64), padding_size=(1024,128)):
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)
        self.reference_dataset = dataset
        self.num_imgs = num_imgs
        self.rescale_mode = rescale_mode
        self.padding_mode = padding_mode
        self.rescale_size = rescale_size
        self.padding_size = padding_size
        if (self.num_imgs > 0) and (self.num_imgs < len(self.reference_dataset)):
            inds = np.random.choice(np.arange(len(self.reference_dataset)),
                                    self.num_imgs, replace=False)
            self.reference_dataset = [self.reference_dataset[i] for i in inds]
        self.meta = meta

    def __getitem__(self, idx):
        data = self.reference_dataset[idx]
        img = cv2.cvtColor(cv2.imread(data['img_path']), cv2.COLOR_BGR2RGB) / 255.
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