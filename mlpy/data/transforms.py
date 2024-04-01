"""TODO"""

import numpy as np

from mlpy.types import Transform


# -------------------------------------------------------------------- One Hot
class OneHot(Transform):
    """TODO"""

    def __init__(self, num_classes: int=1) -> None:
        """TODO"""

        self.num_classes = num_classes

    def __call__(self,
        images: np.ndarray,
        labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """TODO"""

        targets = np.zeros((len(labels), self.num_classes))
        targets[:, labels] = 1

        return images, targets


# ------------------------------------------------------------------ Normalize
class Normalize(Transform):
    """TODO"""

    def __init__(self) -> None:
        """TODO"""

    def __call__(self,
        images: np.ndarray,
        labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """TODO"""

        return images, labels


# --------------------------------------------------------------------- Center
class Center(Transform):
    """TODO"""

    def __init__(self) -> None:
        """TODO"""

    def __call__(self,
        images: np.ndarray,
        labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """TODO"""

        return images, labels


# -------------------------------------------------------------------- Combine
class Combine(Transform):
    """TODO"""

    def __init__(self, transforms: list[Transform]) -> None:
        """TODO"""

        self.transforms = transforms

    def __call__(self,
        images: np.ndarray,
        labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """TODO"""

        for transform in self.transforms:
            images, labels = transform(images, labels)

        return images, labels
