"""
Module containing a range of loss functions commonly used in neural networks
    and machine learning applications.
"""


import numpy as np


class MSE():
    def __init__(self) -> None:
        pass
        
    def forward(self, p, t):
        pass

    def backward(self, p, t):
        pass


def mean_squared_error(prediction, target, derivative=False):
    result = np.square(target - prediction).mean()
    if derivative: result = 2 * (prediction - target) / len(target)
    return result


def categorical_crossentropy(prediction, target, derivative=False):
    result = -np.sum(target * np.log(prediction))
    if derivative: result = prediction - target
    return result