"""
This script loads and processes training data stored in an HDF5 file, using the h5py library. The training data consists of images and is stored in the 'images' dataset of the HDF5 file. The script calculates the mean and standard deviation of the red, green, and blue channels of the images and saves these values to variables named 'train_images_mean' and 'train_images_std' respectively. Finally, the script outputs the number of training samples and the mean and standard deviation values to the console.

"""


import os
import PIL
import h5py

import numpy as np

import torchvision.transforms as tvtrans

from pathlib import Path


# 18000 samples
##  Train est 
##  Val set 
# mean (118.25, 165.38, 92.55) std (40.48, 35.02, 51.05)

main_folder = Path('C:/Users/michele.wiseman/Desktop/Saliency_based_Grape_PM_Quantification-main/data/segmentation/')
dataset_folder = Path('D:/Stacked/Deposition_Study')
train_set_filepath = dataset_folder / 'val_set.hdf5'

# Load data
with h5py.File(train_set_filepath, 'r') as f:
    image_ds = f['images']
    train_images = image_ds[:, ]

train_images_red = train_images[..., 0]
train_images_green = train_images[..., 1]
train_images_blue = train_images[..., 2]

train_images_mean = (np.mean(train_images_red), np.mean(train_images_green), np.mean(train_images_blue))
train_images_std = (np.std(train_images_red), np.std(train_images_green), np.std(train_images_blue))

print(f'{train_images.shape[0]} training samples')
print(f'train mean {train_images_mean} std {train_images_std}')