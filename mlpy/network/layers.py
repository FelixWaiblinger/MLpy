"""Common layers used in neural networks."""

import numpy as np

from mlpy.types import Layer, WeightInit
from mlpy.network.weights import GlorotWeights


# ---------------------------------------------------------------------- Dense
class Dense(Layer):
    def __init__(self,
            features_in: int,
            features_out: int,
            weight_init: WeightInit=GlorotWeights,
            **kwargs
    ) -> None:
        """TODO"""

        super().__init__(**kwargs)
        self.w = weight_init()((features_in, features_out))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """TODO"""

        self.cache = ...

        return x.dot(self.w)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """TODO"""

        # TODO
        return super().backward(x)


# ------------------------------------------------------------- 1D Convolution
class Conv1D(Layer):
    def __init__(self,
            channels_in: int,
            channels_out: int,
            kernel: int,
            weight_init: WeightInit=GlorotWeights,
            **kwargs
    ) -> None:
        """TODO"""

        super().__init__(**kwargs)
        self.w = weight_init()((kernel, channels_out))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """TODO

        Args:
            ``x``: input features from previous layer

        Returns:
            ndarray: feature maps
        """

        # TODO
        return x
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """TODO
        
        Args:
            ``x``: upstream gradient from the next layer

        Returns:
            ndarray: gradient of convolution operation
        """

        # TODO
        return x
    

# ------------------------------------------------------------- 2D Convolution
class Conv2D(Layer):
    def __init__(self,
            channels_in: int,
            channels_out: int,
            kernel: int | tuple[int, int],
            weight_init: WeightInit=GlorotWeights,
            **kwargs
    ) -> None:
        """TODO"""

        super().__init__(**kwargs)
        self.weights = weight_init()((kernel, kernel, channels_out))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """TODO

        Args:
            ``x``: input features from previous layer

        Returns:
            ndarray: feature maps
        """

        # TODO
        return x
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """TODO
        
        Args:
            ``x``: upstream gradient from the next layer

        Returns:
            ndarray: gradient of convolution operation
        """

        # TODO
        return x


# -------------------------------------------------------------------- Sigmoid
class Sigmoid(Layer):
    def __init__(self, **kwargs) -> None:
        """TODO"""

        super().__init__(**kwargs)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the sigmoid function of the input

        Args:
            ``x``: input features from previous layer

        Returns:
            ndarray: element-wise sigmoid of the input
        """

        # TODO consider numerically stable version
        self.cache = 1 / (1 + np.exp(-x))

        return self.cache
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient w.r.t. the inputs.

        Args:
            ``x``: upstream gradient from the next layer

        Returns:
            ndarray: gradient of sigmoid activation function
        """

        return x * self.cache * (1 - self.cache)
    

# ----------------------------------------------------------------------- ReLU
class ReLU(Layer):
    def __init__(self, **kwargs) -> None:
        """TODO"""
        super().__init__(**kwargs)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the ReLU function of the input.

        Args:
            ``x``: input features from previous layer

        Returns:
            ndarray: element-wise ReLU of the input
        """

        self.cache = x

        return np.maximum(np.zeros_like(x), x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient w.r.t. the inputs.

        Args:
            ``x``: upstream gradient from the next layer

        Returns:
            ndarray: gradient of ReLU activation function
        """

        return x * (self.cache > 0)
