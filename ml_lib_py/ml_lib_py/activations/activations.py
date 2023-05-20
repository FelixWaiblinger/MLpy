"""
Module containing a range of activation functions commonly used in neural
    networks and machine learning applications.
"""


import numpy as np


def sigmoid(x, derivative=False):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)

    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])

    dividend = np.ones_like(x)
    dividend[neg_mask] = z[neg_mask]

    result = dividend / (1 + z)

    if derivative: result = result * (1 - result)
    return result


def softmax(x, derivative=False):
    x_shift = x - max(x)
    result = np.exp(x_shift) / np.sum(np.exp(x_shift))
    if derivative: result = np.ones(x.shape) # TODO
    return result


def linear(x, derivative=False):
    return np.ones(x.shape) if derivative else x
