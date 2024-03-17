"""
"""

import os
from os.path import join

import numpy as np
import matplotlib.pyplot as plt

from mlpy.types import Dataset


class ImageFolderDataset(Dataset):
    def __init__(self, path: str, **kwargs) -> None:
        """TODO"""

        super().__init__(**kwargs)
        self.labels = sorted([d.name for d in os.scandir(path) if d.is_dir()])
        self.image_paths = []
        self.images = None

        for label in self.labels:
            for root, _, files in sorted(os.walk(join(path, label))):
                for f in sorted(files):
                    if f.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(join(root, f))

                    if self.in_memory:
                        img = plt.imread(join(root, f))
                        if self.images == None:
                            self.images = img.reshape((1,) + img.shape)
                        else:
                            self.images = np.concatenate(self.images, img)

    def __getitem__(self, index):
        """TODO"""

        if self.in_memory:
            img = self.images[index]
        else:
            img = plt.imread(self.image_paths[index])
            
        label = self.labels[index]

        return {'image': img, 'label': label}
    
    def __len__(self) -> int:
        """TODO"""
        
        return len(self.image_paths)


class CSVDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self) -> int:
        return super().__len__()