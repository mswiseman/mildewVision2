import os
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision


class HyphalDataset(torch.utils.data.Dataset):
    label_class_map = {0: 'clear', 1: 'infected', 2: 'conidiophore'}

    def __init__(self, dataset_path, train=True, transform=None, target_transform=None):
        self.root_dir = Path(dataset_path['root_path'])
        self.train_filepath = self.root_dir / dataset_path['train_filepath']
        self.test_filepath = self.root_dir / dataset_path['test_filepath']
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.data_filepath = self.train_filepath if train else self.test_filepath

        # IMPORTANT: don't open h5 here for images; just read labels once (small)
        with h5py.File(self.data_filepath, "r") as f:
            label_ds = f["labels"]
            self.labels = label_ds[:].reshape(-1).astype(np.int64)
            self._length = len(self.labels)

        # These will be initialized lazily per worker/process
        self._h5 = None
        self._image_ds = None

    def _init_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.data_filepath, "r")
            self._image_ds = self._h5["images"]

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        self._init_h5()

        cur_images = self._image_ds[idx]  # reads one image only (H, W, C)
        y = int(self.labels[idx])  # 0/1/2

        if self.transform is not None:
            cur_images = self.transform(cur_images)

        # map to 2-head multilabel target
        y_infected = 1.0 if y in (1, 2) else 0.0
        y_spor = 1.0 if y == 2 else 0.0
        y_multi = np.array([y_infected, y_spor], dtype=np.float32)

        if self.target_transform is not None:
            y_multi = self.target_transform(y_multi)

        return cur_images, y_multi


def worker_init_fn(worker_id):
    print(torch.utils.data.get_worker_info())


# use to test the dataset class
def test_class():
    #label_class_map = {0: 'clear', 1: 'infected', 2: 'conidiophore'}

    # Parameters for dataset
    dataset_path = {
        'root_path': r'C:\Users\michele.wiseman\Desktop\blackbird_ml\data\\',
        'meta_filepath': 'metadata.csv',
        'train_filepath': 'train.hdf5',
        'test_filepath': 'test.hdf5'
    }

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),  # convert to PIL image
        # torchvision.transforms.Resize(299),  # resize to 299x299
        torchvision.transforms.RandomHorizontalFlip(p=0.5),  # flip horizontally with probability 0.5
        # torchvision.transforms.RandomAffine(degrees=(0, 180), translate=(0.1, 0.1), scale=(0.8, 1.2)), # random affine transformation
        # torchvision.transforms.RandomRotation(degrees=(0, 180)),
        # torchvision.transforms.CenterCrop(224), # crop to 224x224
        torchvision.transforms.ToTensor()  # convert to tensor
    ])

    hyphal_train_ds = HyphalDataset(
        dataset_path, train=True, transform=transform)

    hyphal_dl = torch.utils.data.DataLoader(hyphal_train_ds,
                                            batch_size=4,
                                            shuffle=False,
                                            num_workers=2,
                                            worker_init_fn=worker_init_fn)
    data_iter = iter(hyphal_dl)  # create an iterator for the dataloader
    for i in range(2):  # iterate over the dataloader
        print(images.shape, images.dtype)
        print(labels.shape, labels.dtype)
        print(labels)
        images, labels = next(data_iter)  # get the next batch of images and labels
        f, axarr = plt.subplots(2, 2)
        j = 0
        for row in range(2):
            for col in range(2):
                cur_img = images[j,]
                cur_label = labels[j]
                axarr[row, col].imshow(np.transpose(cur_img, (1, 2, 0)))
                axarr[row, col].set_title(f"I{int(cur_label[0].item())} S{int(cur_label[1].item())}")
                j += 1
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'test_{i}.png')


def main():
    test_class()