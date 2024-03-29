"""TODO"""

import pytest
from mlpy.search import BreadthFirst, UniformCost, GreedyBestFirst, \
    DepthFirst, IterativeDeepening, AStar


@pytest.fixture(scope='session')
def node_structures():
    """TODO"""

    # TODO
    graph = None
    graph_start = None
    graph_end = None
    tree = None
    tree_start = None
    tree_end = None
    grid = None
    grid_start = None
    grid_end = None

    return \
        (graph, graph_start, graph_end), \
        (tree, tree_start, tree_end), \
        (grid, grid_start, grid_end)


# ------------------------------------------------------- Breadth First Search
def test_breadth_first(node_structures):
    """TODO"""

    # unwrap data
    graph, tree, grid = node_structures
    graph, graph_start, graph_end = graph
    tree, tree_start, tree_end = tree
    grid, grid_start, grid_end = grid

    bfs = BreadthFirst()

    graph_path1 = bfs.find(graph_start, graph_end)
    graph_path2 = bfs.find(graph_start, graph_end, max_iters=3)

    tree_path1 = bfs.find(tree_start, tree_end)

    grid_path1 = bfs.find(grid_start, grid_end)

    assert graph_path1
    assert not graph_path2

    assert tree_path1

    assert grid_path1


# -------------------------------------------------------- Uniform Cost Search
def test_uniform_cost(node_structures):
    """TODO"""

    # unwrap data
    graph, tree, grid = node_structures
    graph, graph_start, graph_end = graph
    tree, tree_start, tree_end = tree
    grid, grid_start, grid_end = grid

    ucs = UniformCost()

    graph_path1 = ucs.find(graph_start, graph_end)
    graph_path2 = ucs.find(graph_start, graph_end, max_iters=3)

    tree_path1 = ucs.find(tree_start, tree_end)

    grid_path1 = ucs.find(grid_start, grid_end)

    assert graph_path1
    assert not graph_path2

    assert tree_path1

    assert grid_path1


# --------------------------------------------------- Greedy Best First Search
def test_greedy_best_first(node_structures):
    """TODO"""

    # unwrap data
    graph, tree, grid = node_structures
    graph, graph_start, graph_end = graph
    tree, tree_start, tree_end = tree
    grid, grid_start, grid_end = grid

    gbf = GreedyBestFirst()

    graph_path1 = gbf.find(graph_start, graph_end)
    graph_path2 = gbf.find(graph_start, graph_end, max_iters=3)

    tree_path1 = gbf.find(tree_start, tree_end)

    grid_path1 = gbf.find(grid_start, grid_end)

    assert graph_path1
    assert not graph_path2

    assert tree_path1

    assert grid_path1


# --------------------------------------------------------- Depth First Search
def test_depth_first(node_structures):
    """TODO"""

    # unwrap data
    graph, tree, grid = node_structures
    graph, graph_start, graph_end = graph
    tree, tree_start, tree_end = tree
    grid, grid_start, grid_end = grid

    dfs = DepthFirst()

    graph_path1 = dfs.find(graph_start, graph_end)
    graph_path2 = dfs.find(graph_start, graph_end, max_iters=3)

    tree_path1 = dfs.find(tree_start, tree_end)

    grid_path1 = dfs.find(grid_start, grid_end)

    assert graph_path1
    assert not graph_path2

    assert tree_path1

    assert grid_path1


# ------------------------------------------------- Iterative Deepening Search
def test_iterative_deepening(node_structures):
    """TODO"""

    # unwrap data
    graph, tree, grid = node_structures
    graph, graph_start, graph_end = graph
    tree, tree_start, tree_end = tree
    grid, grid_start, grid_end = grid

    ids = IterativeDeepening()

    graph_path1 = ids.find(graph_start, graph_end)
    graph_path2 = ids.find(graph_start, graph_end, max_iters=3)

    tree_path1 = ids.find(tree_start, tree_end)

    grid_path1 = ids.find(grid_start, grid_end)

    assert graph_path1
    assert not graph_path2

    assert tree_path1

    assert grid_path1


# -------------------------------------------------------------- A Star Search
def test_a_star(node_structures):
    """TODO"""

    # unwrap data
    graph, tree, grid = node_structures
    graph, graph_start, graph_end = graph
    tree, tree_start, tree_end = tree
    grid, grid_start, grid_end = grid

    ast = AStar()

    graph_path1 = ast.find(graph_start, graph_end)
    graph_path2 = ast.find(graph_start, graph_end, max_iters=3)

    tree_path1 = ast.find(tree_start, tree_end)

    grid_path1 = ast.find(grid_start, grid_end)

    assert graph_path1
    assert not graph_path2

    assert tree_path1

    assert grid_path1
