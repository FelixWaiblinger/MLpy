"""
Module containing a range of datasets commonly used in neural networks and
    machine learning applications.
"""


import numpy as np
from mnist import MNIST


def load_mnist(split='both'):
    data = MNIST("datasets/MNIST")

    if split in ['train', 'both']:
        x_train, y_train = data.load_training()
        x_train = np.asarray(x_train).astype(np.float32)[:20000]
        y_train = np.asarray(y_train).astype(np.float32)[:20000]
        y_train = one_hot(y_train, 10)
    
    if split in ['test', 'both']:
        x_test, y_test = data.load_testing()
        x_test = np.asarray(x_test).astype(np.float32)
        y_test = np.asarray(y_test).astype(np.float32)
        y_test = one_hot(y_test, 10)
    
    if split == 'train': return x_train, y_train
    elif split == 'test': return x_test, y_test
    else: return x_train, y_train, x_test, y_test


def one_hot(y, num_classes):
    targets = np.zeros((len(y), num_classes))
    for i, t in enumerate(y):
        targets[i][(int)(t)] = 1
    return targets