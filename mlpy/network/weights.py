"""weight initialization"""

import numpy as np

from mlpy.types import WeightInit


class GlorotWeights(WeightInit):
    def __init__(self) -> None:
        """Glorot uniform weight initialization"""
        pass

    def __call__(self, shape: tuple) -> np.ndarray:
        """TODO
        
        Args:
            ``shape``: shape of weights to be returned

        Returns:
            ndarray: initialized weights as numpy array
        """

        lower = -np.sqrt(6 / sum(shape))
        upper = np.sqrt(6 / sum(shape))

        return np.random.uniform(lower, upper)
    

class XavierWeights(WeightInit):
    def __init__(self) -> None:
        """Xavier weight initialization"""
        pass
        
    def __call__(self, shape: tuple) -> np.ndarray:
        """TODO

        Args:
            ``shape``: shape of weights to be returned

        Returns:
            ndarray: initialized weights as numpy array
        """

        return np.zeros(...)
    

class HeWeights(WeightInit):
    def __init__(self) -> None:
        """He weight initialization"""
        pass
        
    def __call__(self, shape: tuple) -> np.ndarray:
        """TODO

        Args:
            ``shape``: shape of weights to be returned
            
        Returns:
            ndarray: initialized weights as numpy array
        """

        return np.zeros(...)