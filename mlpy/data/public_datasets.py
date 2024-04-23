"""Common datasets used in machine / deep learning applications."""

import os
import gzip
import requests
import numpy as np

from mlpy.types import Dataset, Transform


# -------------------------------------------------------------- MNIST Dataset
class MNIST(Dataset):
    """MNIST dataset containing 70000 samples of 28x28 greyscale images and
    labels for handwritten digits, split into training and test data at a
    ratio of 6/1
    """

    def __init__(self,
        local_path: str=os.path.join('datasets', 'mnist'),
        transform: Transform=None,
        verbose: bool=True
    ) -> None:
        """If necessary, downloads and stores mnist dataset at given local
            path, before loading it into memory.

        Multiple instantiations of this class may lead to high memory
            consumption.

        Args:
            ``local_path``: The path to store / load the dataset to / from

            ``transform``: Transform to apply to the data in memory

            ``verbose``: Printing additional information to the console
        """

        super().__init__(in_memory=True)
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.num_train = 60_000
        self.num_test = 10_000
        self.num_classes = 10

        num_pixels = 28 * 28
        folders = [
            'train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz'
        ]
        mnist_urls = [
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
        ]

        # check for existing data and download if necessary
        if not os.path.exists(local_path):
            os.makedirs(local_path)

        if sorted(os.listdir(local_path)) != sorted(folders):
            if verbose:
                print("Downloading mnist dataset...")

            for i, url in enumerate(mnist_urls):
                # download file
                req = requests.get(url, allow_redirects=True, timeout=20)
                file_path = os.path.join(local_path, folders[i])

                # store file
                with open(file_path, 'wb') as file:
                    file.write(req.content)

        elif verbose:
            print("Found local mnist dataset...")

        if verbose:
            print("Loading dataset into memory...")

        # train images
        with gzip.open(os.path.join(local_path, folders[0]), 'rb') as file:
            file.read(16) # first two bytes are irrelevant
            buffer = file.read(num_pixels * self.num_train)
            data = np.frombuffer(buffer, dtype=np.uint8).astype(float)
            self.x_train = data.reshape((self.num_train, 28, 28, 1))

        # train labels
        with gzip.open(os.path.join(local_path, folders[1]), 'rb') as file:
            file.read(8) # first byte is irrelevant
            buffer = file.read(self.num_train)
            data = np.frombuffer(buffer, dtype=np.uint8).astype(int)
            self.y_train = data.reshape((self.num_train,))

        # test images
        with gzip.open(os.path.join(local_path, folders[2]), 'rb') as file:
            file.read(16) # first two bytes are irrelevant
            buffer = file.read(num_pixels * self.num_test)
            data = np.frombuffer(buffer, dtype=np.uint8).astype(float)
            self.x_test = data.reshape((self.num_test, 28, 28, 1))

        # test labels
        with gzip.open(os.path.join(local_path, folders[3]), 'rb') as file:
            file.read(8) # first byte is irrelevant
            buffer = file.read(self.num_test)
            data = np.frombuffer(buffer, dtype=np.uint8).astype(int)
            self.y_test = data.reshape((self.num_test,))

        del buffer
        del data

        # apply the given transform
        if transform:
            if verbose:
                print("Applying transforms...")

            self.x_train, self.y_train = transform(self.x_train, self.y_train)
            self.x_test, self.y_test = transform(self.x_test, self.y_test)

        print("Finished loading dataset!")

    def __len__(self) -> int:
        """Return overall number of samples (train set + test set)"""
        return self.num_train + self.num_test

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return train / test sample by index (as if the test set were
        concatenated to the train set)
        """
        if self.num_train <= index < self.num_test:
            index -= self.num_train
            return self.x_test[index], self.y_test[index]

        return self.x_train[index], self.y_train[index]
