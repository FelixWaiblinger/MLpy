"""TODO"""

import pytest
from mlpy.search import Graph, Tree, Grid


@pytest.fixture(scope='session')
def node_structures():
    """TODO"""

    # TODO

# ---------------------------------------------------------------------- Graph
def test_graph():
    """TODO"""

    # TODO
    nodes = ...
    edges = ...
    graph = Graph(nodes, edges)

    assert graph

# ----------------------------------------------------------------------- Tree
def test_tree():
    """TODO"""

    # TODO
    root = ...

    tree = Tree(root)

    assert tree

# ----------------------------------------------------------------------- Grid
def test_grid():
    """TODO"""

    # TODO
    size = (5, 5)

    grid = Grid(size)

    assert grid
