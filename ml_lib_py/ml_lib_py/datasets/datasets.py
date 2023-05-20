"""
Module containing a range of datasets commonly used in neural networks and
    machine learning applications.
"""


import os
import gzip
import shutil
import requests
import numpy as np
from mnist import MNIST


def load_mnist(
        split='both',
        encode_labels=True,
        download_path=os.path.join('datasets', 'MNIST')
):
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    mnist_folders = ['t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte',
                     'train-images-idx3-ubyte', 'train-labels-idx1-ubyte']
    if os.listdir(download_path) != mnist_folders:
        _download_mnist(download_path, mnist_folders)
    
    data = MNIST("datasets/MNIST")

    if split in ['train', 'both']:
        x_train, y_train = data.load_training()
        x_train = np.asarray(x_train).astype(np.float32)
        y_train = np.asarray(y_train).astype(np.float32)
        if encode_labels: y_train = one_hot(y_train, 10)
    
    if split in ['test', 'both']:
        x_test, y_test = data.load_testing()
        x_test = np.asarray(x_test).astype(np.float32)
        y_test = np.asarray(y_test).astype(np.float32)
        if encode_labels: y_test = one_hot(y_test, 10)
    
    if split == 'train': return x_train, y_train
    elif split == 'test': return x_test, y_test
    else: return x_train, y_train, x_test, y_test


def _download_mnist(path, folders):
    mnist_urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                  'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                  'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                  'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
    
    for i, url in enumerate(mnist_urls):
        # download files
        r = requests.get(url, allow_redirects=True)
        file = os.path.join(path, folders[i])
        file_gz = file + '.gz'
        
        # store files
        with open(file_gz, 'wb') as f:
            f.write(r.content)

        # unzip files
        with gzip.open(file_gz, 'rb') as f_in, open(file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        os.remove(file_gz)


def one_hot(y, num_classes):
    targets = np.zeros((len(y), num_classes))
    for i, t in enumerate(y):
        targets[i][(int)(t)] = 1
    return targets