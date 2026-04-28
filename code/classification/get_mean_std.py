import os
import PIL
import h5py

import numpy as np

import torchvision.transforms as tvtrans

from pathlib import Path


# February 2023
# Train set 16260
# 
# train mean (81.61065830073957, 155.16163045440445, 118.27642841478439) std (48.42398110804237, 35.264796689262084, 38.50194364001998)

# Entire dataset 

main_folder = Path('C:/Users/michele.wiseman/Desktop/Saliency_based_Grape_PM_Quantification-main/data/')
dataset_folder = main_folder
train_set_filepath = dataset_folder / 'dataset.hdf5'

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