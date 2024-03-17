"""Common datasets used in machine / deep learning applications."""

import os
import gzip
import requests
import numpy as np

from mlpy.types import Transform


class MNISTDataset:
    def __init__(self,
        local_path: str=os.path.join('datasets', 'mnist'),
        transforms: Transform=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """If necessary, downloads and stores mnist dataset at given local
            path, before loading it into memory.

        Multiple instantiations of this class may lead to high memory
            consumption.

        Args:
            ``local_path``: The path to store / load the dataset to / from

        Returns:
            tuple: train and test split of the mnist dataset
        """

        num_train = 50_000
        num_test = 10_000
        image_size = 28
        mnist_folders = [
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
                r = requests.get(url, allow_redirects=True)
                file = os.path.join(path, folders[i])
                file_gz = file + '.gz'
                
                # store file
                with open(file_gz, 'wb') as f:
                    f.write(r.content)

        print("Checking directories...")
        if not os.path.exists(local_path):
            os.makedirs(local_path)

        if sorted(os.listdir(local_path)) != mnist_folders:
            _download_mnist(local_path, mnist_folders)

        print("Loading dataset into memory...")

        # train images
        with gzip.open(os.path.join(local_path, mnist_folders[0]), 'rb') as f:
            f.read(16) # first two bytes are irrelevant
            buffer = f.read(image_size * image_size * num_train)
            data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
            x_train = data.reshape(num_train, image_size, image_size, 1)

        # train labels
        with gzip.open(os.path.join(local_path, mnist_folders[1]), 'rb') as f:
            f.read(8) # first byte is irrelevant
            buffer = f.read(num_train)
            data = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)
            y_train = data.reshape(num_train, image_size, image_size, 1)

        # test images
        with gzip.open(os.path.join(local_path, mnist_folders[2]), 'rb') as f:
            f.read(16) # first two bytes are irrelevant
            buffer = f.read(image_size * image_size * num_test)
            data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
            x_test = data.reshape(num_test, image_size, image_size, 1)

        # test labels
        with gzip.open(os.path.join(local_path, mnist_folders[3]), 'rb') as f:
            f.read(8) # first byte is irrelevant
            buffer = f.read(image_size * image_size * num_test)
            data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
            y_test = data.reshape(num_test, image_size, image_size, 1)

        # TODO apply transforms
        if transforms is not None:
            print("Applying transforms...")
        
        print("Successfully loaded dataset!")
        return x_train, y_train, x_test, y_test
