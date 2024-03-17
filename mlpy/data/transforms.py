"""TODO"""

import numpy as np


def one_hot(
    targets: np.ndarray,
    num_classes: int
) -> np.ndarray:
    """TODO"""

    new_targets = np.zeros((len(targets), num_classes))
    new_targets[:, targets] = 1 # TODO test

    return new_targets