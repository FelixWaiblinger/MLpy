"""Transforms"""

import numpy as np

from mlpy.types import Transform


# -------------------------------------------------------------------- One Hot
class OneHot(Transform):
    """One-hot encoding for classification labels"""

    def __init__(self, num_classes: int=1) -> None:
        """Transform label vector of shape [num_samples,] to matrix of shape
        [num_samples, num_classes] where the (i, j)-th element indicates the
        i-th sample being of class j

        Args:
            ``num_classes``: number of classes available in this task
        """

        self.num_classes = num_classes

    def __call__(self,
        images: np.ndarray,
        labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform one-hot encoding on label vector
        
        Args:
            ``images``: input features
                shape: [num_samples, width, height, channels]

            ``labels``: target values
                shape: [num_samples,]
        """

        labels = np.eye(self.num_classes)[labels]

        return images, labels


# ------------------------------------------------------------------ Normalize
class Normalize(Transform):
    """Normalization of the input features"""

    def __init__(self, target_range: tuple[float, float]=(0, 1)) -> None:
        """Transform the values of the input features to be in the given range
        by dividing every value by the largest one in the data

        Args:
            ``target_range``: a range of values to shift the data into
                default: [0, 1]
        """

        self.low, self.high = target_range

    def __call__(self,
        images: np.ndarray,
        labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform normalization on input features
        
        Args:
            ``images``: input features
                shape: [num_samples, width, height, channels]

            ``labels``: target values
                shape: [num_samples,]
        """

        old_low = np.min(images)
        old_high = np.max(images)
        factor = (self.high - self.low) / (old_high - old_low)

        images -= old_low
        images *= factor
        images += self.low

        return images, labels


# --------------------------------------------------------------------- Center
class Center(Transform):
    """Centering of the input features"""

    def __init__(self) -> None:
        """TODO"""

    def __call__(self,
        images: np.ndarray,
        labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform centering of the input features
        
        Args:
            ``images``: input features
                shape: [num_samples, width, height, channels]

            ``labels``: target values
                shape: [num_samples,]
        """

        images -= np.mean(images, axis=(0, 1, 2))[:, np.newaxis]

        return images, labels


# ------------------------------------------------------------------ GrayScale
class GrayScale(Transform):
    """Convert multi-channel images to grayscale"""

    def __init__(self) -> None:
        """TODO"""

    def __call__(self,
        images: np.ndarray,
        labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform conversion of multi-channel images to grayscale
        
        Args:
            ``images``: input features
                shape: [num_samples, width, height, channels]

            ``labels``: target values
                shape: [num_samples,]
        """

        images = np.mean(images, axis=-1)

        return images, labels


# -------------------------------------------------------------------- Combine
class Combine(Transform):
    """Combination of multiple transforms"""

    def __init__(self, transforms: list[Transform]) -> None:
        """TODO"""

        self.transforms = transforms

    def __call__(self,
        images: np.ndarray,
        labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform input features and labels by applying all transformations
        given
        
        Args:
            ``images``: input features
                shape: [num_samples, width, height, channels]

            ``labels``: target values
                shape: [num_samples,]
        """

        for transform in self.transforms:
            images, labels = transform(images, labels)

        return images, labels
