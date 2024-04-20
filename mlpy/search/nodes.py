"""TODO"""

from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from mlpy.types import Node


COLORS = {
    '': 'none',
    'start': 'tab:green',
    'path_start': 'tab:blue',
    'end': 'tab:red',
    'path_end': 'tab:purple',
    'path_': 'tab:cyan',
    'blocked': 'tab:gray'
}

# ---------------------------------------------------------------------- Graph
class Graph:
    """Node-based graph structure"""

    def __init__(self,
        nodes: List[Node],
        edges: List[tuple[Node, Node]], # TODO check: references / extra nodes
        directed: bool=True
    ) -> None:
        """A collection of nodes, connected by (un-)directed edges
        
        Args:
            ``nodes``: Collection of nodes in this graph

            ``edges``: Collection of connected pairs of nodes 

            ``directed``: Whether edges in this graph are directed
        """

        self.nodes = nodes
        self.edges = edges

        # add all edges connecting from end to start aswell
        if not directed:
            self.edges += list(map(lambda x: (x[1], x[0]), self.edges))

    @property
    def is_connected(self) -> bool:
        """Return whether this graph contains more than one subgraph"""

        return False # TODO

    @property
    def is_loop_free(self) -> bool:
        """Return whether this graph contains atleast one loop"""

        visited = []
        for start, end in self.edges:
            if start in visited:
                return False
            visited.append(end)
        return True

    def as_tree(self) -> None:
        """Return this graph as tree if valid"""

        if not self.is_loop_free:
            return None

        return None # TODO


# ----------------------------------------------------------------------- Tree
class Tree:
    """Node-based tree structure"""

    def __init__(self, root: Node) -> None:
        """Instantiate tree from its root
        
        Args:
            ``root``: Root node of this tree
        """

        self.root: Node = root

    @property
    def effective_branching_factor(self) -> float:
        """Return the average number of children in this tree"""

        nodes = []
        self._get_all_nodes(self.root, nodes)

        return np.mean([len(n.neighbors) for n in nodes])

    @property
    def max_depth(self) -> int:
        """Return depth of deepest node in this tree"""

        return max([t.max_depth for t in self.root.neighbors]) + 1

    def as_graph(self) -> Graph:
        """Return this tree represented as a graph"""

        nodes, edges = [], []
        self._get_all_nodes(self.root, nodes)

        for node in nodes:
            for n in node.neighbors:
                edges.append((node, n))

        return Graph(nodes, edges)

    def _get_all_nodes(self, node, storage):
        for child in node.neighbors:
            storage.append(child)
            self._get_all_nodes(child, storage)


# ----------------------------------------------------------------------- Grid
class Grid:
    """Node-based 2D grid structure"""

    def __init__(self,
        size: tuple[int, int],
        diagonal: bool=False,
        blocked: np.ndarray | None=None
    ) -> None:
        """Create a 2D grid with the given dimension initialized with empty
        nodes and movement along given directions
        
        Args:
            ``size``: Shape of this grid

            ``diagonal``: Whether diagonal movement is allowed
        """

        w, h = range(size[0]), range(size[1])
        self.nodes: np.ndarray = np.array([[Node() for _ in w] for _ in h])
        if blocked is None:
            blocked = np.zeros_like(self.nodes, dtype=bool)

        for i in range(size[0]):
            for j in range(size[1]):
                # blocked cells
                if blocked[i, j]:
                    self.nodes[i, j].info['type'] = 'blocked'
                    continue

                # add upper neighbor
                if i > 0 and not blocked[i-1, j]:
                    self.nodes[i, j].neighbors.append(self.nodes[i-1, j])
                # add right neighbor
                if j < size[1]-1 and not blocked[i, j+1]:
                    self.nodes[i, j].neighbors.append(self.nodes[i, j+1])
                # add lower neighbor
                if i < size[0]-1 and not blocked[i+1, j]:
                    self.nodes[i, j].neighbors.append(self.nodes[i+1, j])
                # add left neighbor
                if j > 0 and not blocked[i, j-1]:
                    self.nodes[i, j].neighbors.append(self.nodes[i, j-1])

                if not diagonal:
                    continue

                # add upper right neighbor
                if i > 0 and j < size[1]-1 and not blocked[i-1, j+1]:
                    self.nodes[i, j].neighbors.append(self.nodes[i-1, j+1])
                # add lower right neighbor
                if i < size[0]-1 and j < size[1]-1 and not blocked[i+1, j+1]:
                    self.nodes[i, j].neighbors.append(self.nodes[i+1, j+1])
                # add lower left neighbor
                if i < size[0]-1 and j > 0 and not blocked[i+1, j-1]:
                    self.nodes[i, j].neighbors.append(self.nodes[i+1, j-1])
                # add top left neighbor
                if i > 0 and j > 0 and not blocked[i-1, j-1]:
                    self.nodes[i, j].neighbors.append(self.nodes[i-1, j-1])

    def __getitem__(self, key: int | tuple[int, int]) -> Node:
        """Return the node at the given coordinates
        
        Args:
            ``key``: grid coordinates
        """

        idx = key if isinstance(key, tuple) else (key, key)

        return self.nodes[idx]

    @property
    def size(self) -> tuple[int, int]:
        """Return the size of this grid"""

        return self.nodes.shape

    def set_start(self, key: int | tuple[int, int] | Node) -> None:
        """Set a node as start node
        
        Args:
            ``key``: grid coordinates or a reference to the node itself
        """

        node = key if isinstance(key, Node) else self[key]
        node.info['type'] = 'start'

    def set_end(self, key: int | tuple[int, int] | Node) -> None:
        """Set a node as goal node
        
        Args:
            ``key``: grid coordinates or a reference to the node itself
        """

        node = key if isinstance(key, Node) else self[key]
        node.info['type'] = 'end'

    def show(self, path: List[Node] | None=None) -> None:
        """Plot this 2D grid with additional informations as specified in each
        node
        """

        width, height = self.size
        kwargs = {'linewidth': 2, 'edgecolor': 'k'}
        _, ax = plt.subplots()

        for i in range(width):
            for j in range(height):
                # assign correct colors
                node_type = self[i, j].info.get('type', '')
                if path and self[i, j] in path:
                    node_type = 'path_' + node_type

                kwargs['facecolor'] = COLORS[node_type]
                ax.add_patch(Rectangle((i, j), 1, 1, **kwargs))

        plt.xticks(range(width+1), range(width+1))
        plt.yticks(range(height+1), range(height+1))
        plt.show()


# ------------------------------------------------ Some Continuous Environment
# TODO
