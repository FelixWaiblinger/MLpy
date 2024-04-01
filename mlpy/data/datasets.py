"""Datasets"""

import os
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

from mlpy.types import Dataset


# ------------------------------------------------------- Image Folder Dataset
class ImageFolderDataset(Dataset):
    """Dataset of images organized in a local directory structure"""

    def __init__(self, path: str, **kwargs) -> None:
        """Instance of an Image folder dataset with loaded image paths and
        images optionally stored in memory
        """

        super().__init__(**kwargs)
        self.labels = sorted([d.name for d in os.scandir(path) if d.is_dir()])
        self.image_paths = []
        self.images = None

        for label in self.labels:
            for root, _, files in sorted(os.walk(join(path, label))):
                for file in sorted(files):
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(join(root, file))

                    if self.in_memory:
                        img = plt.imread(join(root, file))
                        if not self.images:
                            self.images = img.reshape((1,) + img.shape)
                        else:
                            self.images = np.concatenate(self.images, img)

    def __getitem__(self, index) -> dict:
        """Return image at given index"""

        if self.in_memory:
            img = self.images[index]
        else:
            img = plt.imread(self.image_paths[index])

        label = self.labels[index]

        return {'image': img, 'label': label}

    def __len__(self) -> int:
        """Return number of images in this dataset"""

        return len(self.image_paths)


# ---------------------------------------------------------------- CSV Dataset
class CSVDataset(Dataset):
    """Dataset of tabular data organized as comma separated values"""

    def __init__(self, path: str, **kwargs) -> None:
        """Instance of a CSV dataset with loaded csv data path and table
        optionally stored in memory
        """

        super().__init__(**kwargs)
        self.data_path = path
        self.data = None
        self._load_args = {'dtype': np.float32, 'delimiter': ','}

        if self.in_memory:
            self.data = np.loadtxt(path, **self._load_args)

    def __getitem__(self, index) -> np.ndarray:
        """Return the index-th row of this dataset"""

        if self.data:
            return self.data[index]

        return np.loadtxt(self.data_path, **self._load_args)

    def __len__(self) -> int:
        """Return number of rows in this dataset"""

        if self.data:
            return len(self.data)

        num_samples = 0
        with open(self.data_path, 'r', encoding='utf-8') as file:
            for _ in file:
                num_samples += 1

        return num_samples
