"""Common evaluation metrics used in machine / deep learning applications."""

import numpy as np


def ACC(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """TODO
    
    Args:
        ``targets``: TODO
        ``predictions``: TODO
    
    Returns:
        float: TODO
    """
    
    correct = 0
    for p, t in zip(predictions, targets):
        if p == t: correct += 1

    return (float)(correct) / len(predictions)


def MSE(
    targets: np.ndarray,
    predictions: np.ndarray
) -> float:
    """TODO
    
    Args:
        ``targets``: TODO
        ``predictions``: TODO
        
    Returns:
        float: TODO
    """

    return np.mean(np.square(targets - predictions))