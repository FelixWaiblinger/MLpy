"""
Module containing a range of layers commonly used in neural networks and
    similar applications.
"""


from abc import ABC, abstractmethod

import numpy as np


# ----------------------------------------------------------------- Base layer
class Layer(ABC):
    def __init__(self):
        """initialize this layer"""
        self.cache = None

    @abstractmethod
    def forward(self, x):
        """perform the forward pass in this layer"""

    @abstractmethod
    def backward(self, x):
        """perform the backward pass in this layer"""


# ------------------------------------------------------ Fully-connected layer
class Dense(Layer):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.w = np.zeros((num_in, num_out))
        self._init_weights()

    # glorot uniform weight initialization
    def _init_weights(self):
        self.w = np.random.uniform(
            -np.sqrt(6 / sum(self.w.shape)),
            np.sqrt(6 / sum(self.w.shape))
        )

    def forward(self, x):
        return x.dot(self.w)
    
    def backward(self, x):
        return super().backward(x)


# ----------------------------------------------------- 1D Convolutional layer
class Conv1D(Layer):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.weights = np.zeros((num_in, num_out))
        self._init_weights()

    def _init_weights(self):
        pass

    def forward(self, x):
        return super().forward(x)
    
    def backward(self, x):
        return super().backward(x)
    

# ----------------------------------------------------- 2D Convolutional layer
class Conv2D(Layer):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.weights = np.zeros((num_in, num_out))
        self._init_weights()

    def _init_weights(self):
        pass

    def forward(self, x):
        return super().forward(x)
    
    def backward(self, l):
        return super().backward(l)


# --------------------------------------------------------- Sigmoid activation
class Sigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Compute the sigmoid function of the input\n
            ``x``   numpy output from the previous layer
        """
        self.cache = x
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x):
        """Compute the gradient w.r.t. the input\n
            ``x``   upstream gradient from the next layer
        """
        return x * self.cache * (1 - self.cache)
    

# ------------------------------------------------------------ ReLU activation
class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Compute the sigmoid function of the input\n
            ``x``   numpy output from the previous layer
        """
        self.cache = x
        return np.maximum(np.zeros_like(x), x)
    
    def backward(self, x):
        """Compute the gradient w.r.t. the input\n
            ``x``   upstream gradient from the next layer
        """
        return x * (self.cache > 0)


# TODO cleanup
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
