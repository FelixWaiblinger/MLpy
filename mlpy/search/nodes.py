"""TODO"""

from typing import List

import numpy as np

from mlpy.types import Node


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
        diagonal: bool=False
    ) -> None:
        """Create a 2D grid with the given dimension initialized with empty
        nodes and movement along given directions
        
        Args:
            ``size``: Shape of this grid

            ``diagonal``: Whether diagonal movement is allowed
        """

        self.nodes: np.ndarray = np.full(size, fill_value=Node(), dtype=Node)

        for i in range(size[0]):
            for j in range(size[1]):
                if i > 0: # add upper neighbor
                    self.nodes[i, j].neighbors.append(self.nodes[i-1, j])
                if j < size[1]-1: # add right neighbor
                    self.nodes[i, j].neighbors.append(self.nodes[i, j+1])
                if i < size[0]-1: # add lower neighbor
                    self.nodes[i, j].neighbors.append(self.nodes[i+1, j])
                if j > 0: # add left neighbor
                    self.nodes[i, j].neighbors.append(self.nodes[i, j-1])

                if not diagonal:
                    continue

                if i > 0 and j < size[1]-1: # add upper right neighbor
                    self.nodes[i, j].neighbors.append(self.nodes[i-1, j+1])
                if i < size[0]-1 and j < size[1]-1: # add lower right neighbor
                    self.nodes[i, j].neighbors.append(self.nodes[i+1, j+1])
                if i < size[0]-1 and j > 0: # add lower left neighbor
                    self.nodes[i, j].neighbors.append(self.nodes[i+1, j-1])
                if i > 0 and j > 0: # add top left neighbor
                    self.nodes[i, j].neighbors.append(self.nodes[i-1, j-1])

    @property
    def size(self) -> tuple[int, int]:
        """Return the size of this grid"""
        return self.nodes.shape
