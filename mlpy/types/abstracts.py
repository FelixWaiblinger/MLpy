"""Abstract base types to be implemented by concrete classes."""

from abc import ABC, abstractmethod
from typing import Any, List, Dict


# -------------------------------------------------------------------- Dataset
class Dataset(ABC):
    """Abstract dataset class"""

    def __init__(self, in_memory: bool=False) -> None:
        """Instantiate dataset"""
        self.in_memory = in_memory

    @abstractmethod
    def __getitem__(self, index):
        """Return sample at index"""

    @abstractmethod
    def __len__(self) -> int:
        """Return size of dataset"""


# ------------------------------------------------------------------ Transform
class Transform(ABC):
    """Abstract transform class"""

    def __init__(self) -> None:
        """Instantiate transform"""

    @abstractmethod
    def __call__(self, images, labels):
        """Apply transform"""


# ----------------------------------------------------------------- Dataloader
class Dataloader(ABC):
    """Abstract dataloader class"""

    def __init__(self, dataset: Dataset) -> None:
        """Instantiate dataloader"""


# ----------------------------------------------------------------- Sequential
class Network(ABC):
    """Abstract neural network"""

    def __init__(self, model_name="model") -> None:
        """Instantiate network"""
        self.name = model_name
        self.trainable = True

    @abstractmethod
    def forward(self, x):
        """Compute the forward pass through the network"""

    @abstractmethod
    def backward(self, x):
        """Compute the backward pass through the network"""

    @abstractmethod
    def save_model(self, path: str):
        """Save model architecture and parameters to disk"""

    @abstractmethod
    def load_model(self, path: str):
        """Load model architecture and parameters from disk"""


# ---------------------------------------------------------------------- Layer
class Layer(ABC):
    """Abstract base layer"""

    def __init__(self, layer_name="layer") -> None:
        """Instantiate layer"""
        self.name = layer_name
        self.cache = None

    @abstractmethod
    def forward(self, x):
        """Perform the forward pass in this layer"""

    @abstractmethod
    def backward(self, x):
        """Perform the backward pass in this layer"""


# ----------------------------------------------------------------------- Loss
class Loss(ABC):
    """Abstract base loss"""

    def __init__(self) -> None:
        """Instantiate loss"""

    @abstractmethod
    def forward(self, targets, predictions):
        """Compute the loss"""

    @abstractmethod
    def backward(self, targets, predictions):
        """Compute the gradient of the loss"""


# ----------------------------------------------------------------- WeightInit
class WeightInit(ABC):
    """Abstract weight initialization"""

    def __init__(self) -> None:
        """Instantiate weight initialization"""

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any):
        """Initialize weights"""


# ----------------------------------------------------------------------- Node
class Node(ABC):
    """Abstract traversable node"""

    def __init__(self) -> None:
        """Instantiate node"""
        self.neighbors: List = []
        self.depth: int = 0
        self.cost: float = 1
        self.heuristic: float = 1
        self.info: Dict = {}


# ----------------------------------------------------------------------- Node
class Search(ABC):
    """Abstract search algorithm"""

    def __init__(self, queue_type) -> None:
        """Instantiate search algorithm"""
        self.path: List = []
        self.visited: Dict = {}
        self.frontier = queue_type

    @abstractmethod
    def find(self, start, end, max_iters=10000) -> List[Node]:
        """Abstract method to find a path from start to end"""

    @abstractmethod
    def show(self, start, end, max_iters=10000) -> List[Node]:
        """Abstract method to find a path from start to end, visualizing the
        process
        """

    def _step_callback(self):
        """Callback function after each node expansion"""

    def _backtrack(self, start, end) -> List[Node]:
        """Backtrack from end to start after the goal node has been found"""
        self.path = [end]
        previous = end
        while start not in self.path:
            self.path.append(self.visited[previous])
            previous = self.visited[previous]
        self.path.reverse()
        return self.path
