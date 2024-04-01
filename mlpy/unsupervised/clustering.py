"""TODO"""

import numpy as np

from mlpy.evaluation.metrics import MSE


# --------------------------------------------------------- K-Nearest-neighbor
class KNearestNeighbor:
    """TODO"""

    def __init__(self,
        k: int,
        distance_func: callable=MSE,
        weighted: bool=False
    ) -> None:
        """TODO"""

        self.k = k
        self.distance = distance_func
        self.weighted = weighted
        self.points = None

    def fit(self,
        x: np.ndarray,
        clusters: int
    ) -> None:
        """TODO

        Args:
            ``x``: 
            ``clusters``: 
        """

        self.points = x

    def predict(self, ) -> None:
        """TODO"""
