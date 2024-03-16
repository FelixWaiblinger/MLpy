"""Common loss functions used in neural networks and machine learning
    applications.
"""

import numpy as np

from mlpy.types import Loss


# --------------------------------------------------------- Mean Squared Error
class MSE(Loss):
    def __init__(self) -> None:
        pass
        
    def forward(self,
        targets: np.ndarray,
        predictions: np.ndarray
    ) -> float:
        """TODO"""

        return np.mean(np.square(targets - predictions))

    def backward(self,
        targets: np.ndarray,
        predictions: np.ndarray
    ) -> np.ndarray:
        """TODO"""

        return 2 * (predictions - targets) / len(targets)


# ------------------------------------------------------- Binary Cross Entropy
class BCE(Loss):
    def __init__(self) -> None:
        """Binary cross-entropy"""
        pass

    def forward(self,
        targets: np.ndarray,
        predictions: np.ndarray
    ) -> float:
        """TODO"""
    
        return -np.sum(targets * np.log(predictions))
    
    def backward(self,
        targets: np.ndarray,
        predictions: np.ndarray
    ) -> np.ndarray:
        """TODO"""

        return targets / predictions
    

# -------------------------------------------------- Categorical Cross Entropy
class CCE(Loss):
    def __init__(self) -> None:
        """Categorical cross-entropy"""
        pass

    def forward(self,
        targets: np.ndarray,
        predictions: np.ndarray
    ) -> float:
        """TODO"""
    
        return -np.sum(targets * np.log(predictions))
    
    def backward(self,
        targets: np.ndarray,
        predictions: np.ndarray
    ) -> np.ndarray:
        """TODO"""

        return targets / predictions