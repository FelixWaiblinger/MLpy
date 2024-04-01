"""Common datasets used in machine / deep learning applications."""

import os
import gzip
import requests
import numpy as np

from mlpy.types import Dataset, Transform


# -------------------------------------------------------------- MNIST Dataset
class MNISTDataset(Dataset):
    """MNIST dataset containing 70000 samples of 28x28 greyscale images and
    labels for handwritten digits, split into training and test data at a
    ratio of 6/1
    """

    def __init__(self,
        local_path: str=os.path.join('datasets', 'mnist'),
        transforms: Transform=None
    ) -> None:
        """If necessary, downloads and stores mnist dataset at given local
            path, before loading it into memory.

        Multiple instantiations of this class may lead to high memory
            consumption.

        Args:
            ``local_path``: The path to store / load the dataset to / from

        Returns:
            tuple: train and test split of the mnist dataset
        """

        super().__init__(in_memory=True)
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        num_train = 60_000
        num_test = 10_000
        image_size = 28
        folders = [
            'train-images-idx3-ubyte',
            'train-labels-idx1-ubyte',
            't10k-images-idx3-ubyte',
            't10k-labels-idx1-ubyte'
        ]

        def _download_mnist(path, folders):
            print("Downloading mnist dataset...")

            mnist_urls = [
                'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
            ]

            for i, url in enumerate(mnist_urls):
                # download file
                req = requests.get(url, allow_redirects=True, timeout=20)
                file_path = os.path.join(path, folders[i])
                file_gz = file_path + '.gz'

                # store file
                with open(file_gz, 'wb') as file:
                    file.write(req.content)

        print("Checking directories...")
        if not os.path.exists(local_path):
            os.makedirs(local_path)

        if sorted(os.listdir(local_path)) != folders:
            _download_mnist(local_path, folders)

        print("Loading dataset into memory...")

        # train images
        with gzip.open(os.path.join(local_path, folders[0]), 'rb') as file:
            file.read(16) # first two bytes are irrelevant
            buffer = file.read(image_size * image_size * num_train)
            data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
            self.x_train = data.reshape(num_train, image_size, image_size, 1)

        # train labels
        with gzip.open(os.path.join(local_path, folders[1]), 'rb') as file:
            file.read(8) # first byte is irrelevant
            buffer = file.read(num_train)
            data = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)
            self.y_train = data.reshape(num_train, image_size, image_size, 1)

        # test images
        with gzip.open(os.path.join(local_path, folders[2]), 'rb') as file:
            file.read(16) # first two bytes are irrelevant
            buffer = file.read(image_size * image_size * num_test)
            data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
            self.x_test = data.reshape(num_test, image_size, image_size, 1)

        # test labels
        with gzip.open(os.path.join(local_path, folders[3]), 'rb') as file:
            file.read(8) # first byte is irrelevant
            buffer = file.read(image_size * image_size * num_test)
            data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
            self.y_test = data.reshape(num_test, image_size, image_size, 1)

        # TODO apply transforms
        if transforms is not None:
            print("Applying transforms...")

        print("Successfully loaded dataset!")
