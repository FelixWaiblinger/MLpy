"""
"""

from abc import ABC, abstractmethod
from typing import Any


# -------------------------------------------------------------------- Dataset
class Dataset(ABC):
    def __init__(self, in_memory: bool=False) -> None:
        """Abstract dataset class."""
        self.in_memory = in_memory

    @abstractmethod
    def __getitem__(self, index):
        """Return sample at index."""

    @abstractmethod
    def __len__(self) -> int:
        """Return size of dataset."""


# ------------------------------------------------------------------ Transform
class Transform(ABC):
    def __init__(self) -> None:
        """Abstract transform class."""
    
    @abstractmethod
    def __call__(self, images, labels):
        """TODO"""


# ----------------------------------------------------------------- Dataloader
class Dataloader(ABC):
    def __init__(self) -> None:
        """Abstract dataloader class."""


# ----------------------------------------------------------------- Sequential
class Network(ABC):
    def __init__(self, model_name="model") -> None:
        """Abstract base sequential neural network."""
        self.name = model_name
        self.trainable = True

    @abstractmethod
    def forward(self, x):
        """TODO"""

    @abstractmethod
    def backward(self, x):
        """TODO"""
    
    @abstractmethod
    def save_model(self):
        """TODO"""

    @abstractmethod
    def load_model(self):
        """TODO"""


# ---------------------------------------------------------------------- Layer
class Layer(ABC):
    def __init__(self, layer_name="layer"):
        """Abstract base layer."""
        self.name = layer_name
        self.cache = None

    @abstractmethod
    def forward(self, x):
        """Perform the forward pass in this layer."""

    @abstractmethod
    def backward(self, x):
        """Perform the backward pass in this layer."""


# ----------------------------------------------------------------------- Loss
class Loss(ABC):
    def __init__(self) -> None:
        """Abstract base loss."""

    @abstractmethod
    def forward(self, targets, predictions):
        """Compute the loss."""

    @abstractmethod
    def backward(self, targets, predictions):
        """Compute the gradient of the loss."""


# ----------------------------------------------------------------- WeightInit
class WeightInit(ABC):
    def __init__(self) -> None:
        """Abstract weight initialization method."""

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Initialize weights"""