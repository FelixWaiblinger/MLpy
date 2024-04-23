"""Pathfinding TODO"""

from typing import List

from queue import Queue, LifoQueue, PriorityQueue
from mlpy.types import Node, Search, MAX_INT


# ------------------------------------------------------- Breadth First Search
class BreadthFirst(Search):
    """Breadth First Search Algorithm"""

    def __init__(self) -> None:
        """Instance of the breadth first search algorithm for pathfinding in
        node based environments
        """

        super().__init__(queue_type=Queue())

    def find(self,
        start: Node,
        end: Node,
        max_iters: int=10000
    ) -> List[Node]:
        """Use the breadth first search algorithm to find the path from a
        start node to the goal node, limited by a maximum number of nodes to
        check

        Args:
            ``start``: Initial node to start the search from
            
            ``end``: Goal node to be searched for
            
            ``max_iters``: Maximum number of nodes to be visited
                (default: 10000)

        Returns:
            list: path to goal node or an empty list if goal was not reached
        """

        self.frontier.put(start)
        self.visited[start] = None

        # start node is goal node
        if start == end:
            return [start]

        # limit loops and stop if no more nodes available
        while not self.frontier.empty():
            max_iters -= 1
            node = self.frontier.get()

            # check neighbors
            for child in node.neighbors:
                # avoid infinite loops
                if child in self.visited:
                    continue

                self.visited[child] = node

                # goal node found
                if child == end:
                    return self._backtrack(start, end)

                # extend search
                self.frontier.put(child)

            # limit number of nodes explored
            if max_iters < 1:
                break

        # no goal found (within maximum iterations)
        return []

    def show(self, start, end, max_iters=10000) -> None:
        """TODO"""


# -------------------------------------------------------- Uniform Cost Search
class UniformCost(Search):
    """Uniform Cost Search Algorithm"""

    def __init__(self) -> None:
        """Instance of the uniform cost search algorithm for pathfinding in
        node based environments
        """

        super().__init__(queue_type=PriorityQueue())

    def find(self,
        start: Node,
        end: Node,
        max_iters: int=10000
    ) -> List[Node]:
        """Use the uniform cost search algorithm to find the path from a start
        node to the goal node, prioritized by a path cost value and limited by
        a maximum number of nodes to check

        Args:
            ``start``: Initial node to start the search from
            
            ``end``: Goal node to be searched for
            
            ``max_iters``: Maximum number of nodes to be visited
                (default: 10000)

        Returns:
            list: path to goal node or an empty list if goal was not reached
        """

        start.info['priority'] = start.cost
        self.frontier.put(start)
        self.visited[start] = None

        # start node is goal node
        if start == end:
            return [start]

        # limit loops and stop if no more nodes available
        while not self.frontier.empty():
            max_iters -= 1
            node = self.frontier.get()

            # check neighbors
            for child in node.neighbors:
                # avoid infinite loops
                if child in self.visited:
                    continue

                self.visited[child] = node

                # goal node found
                if child == end:
                    return self._backtrack(start, end)

                # extend search
                child.info['priority'] = child.cost
                self.frontier.put(child)

            # limit number of nodes explored
            if max_iters < 1:
                break

        # no goal found (within maximum iterations)
        return []

    def show(self, start, end, max_iters=10000) -> None:
        """TODO"""


# --------------------------------------------------- Greedy Best First Search
class GreedyBestFirst(Search):
    """Greedy Best First Search Algorithm"""

    def __init__(self) -> None:
        """Instance of the greedy best first search algorithm for pathfinding
        in node based environments
        """

        super().__init__(queue_type=PriorityQueue())

    def find(self,
        start: Node,
        end: Node,
        max_iters: int=10000
    ) -> List[Node]:
        """Use the greedy best first search algorithm to find the path from a
        start node to the goal node, prioritized by a heuristic value and
        limited by a maximum number of nodes to check

        Args:
            ``start``: Initial node to start the search from
            
            ``end``: Goal node to be searched for
            
            ``max_iters``: Maximum number of nodes to be visited
                (default: 10000)

        Returns:
            list: path to goal node or an empty list if goal was not reached
        """

        start.info['priority'] = start.heuristic
        self.frontier.put(start)
        self.visited[start] = None

        # start node is goal node
        if start == end:
            return [start]

        # limit loops and stop if no more nodes available
        while not self.frontier.empty():
            max_iters -= 1
            node = self.frontier.get()

            # check neighbors
            for child in node.neighbors:
                # avoid infinite loops
                if child in self.visited:
                    continue

                self.visited[child] = node

                # goal node found
                if child == end:
                    return self._backtrack(start, end)

                # extend search
                child.info['priority'] = child.heuristic
                self.frontier.put(child)

            # limit number of nodes explored
            if max_iters < 1:
                break

        # no goal found (within maximum iterations)
        return []

    def show(self, start, end, max_iters=10000) -> None:
        """TODO"""


# --------------------------------------------------------- Depth First Search
class DepthFirst(Search):
    """Depth First Search Algorithm"""

    def __init__(self, max_depth: int=MAX_INT) -> None:
        """Instance of the depth first search algorithm for pathfinding in
        node based environments

        Args:
            ``max_depth``: Optional depth limit to do Depth Limited Search
        """

        super().__init__(queue_type=LifoQueue())
        self.max_depth = max_depth

    def find(self,
        start: Node,
        end: Node,
        max_iters: int=10000
    ) -> List[Node]:
        """Use the depth first search algorithm to find the path from a start
        node to the goal node, limited by a maximum number of nodes to check

        Args:
            ``start``: Initial node to start the search from
            
            ``end``: Goal node to be searched for
            
            ``max_iters``: Maximum number of nodes to be visited
                (default: 10000)

        Returns:
            list: path to goal node or an empty list if goal was not reached
        """

        self.frontier.put(start)
        self.visited[start] = None

        # start node is goal node
        if start == end:
            return [start]

        # stop if no more nodes are available
        while not self.frontier.empty():
            max_iters -= 1
            node = self.frontier.get()

            # only explore nodes up to the current max depth
            if node.depth > self.max_depth:
                continue

            # check neighbors
            for child in node.neighbors:
                # avoid infinite loops
                if child in self.visited:
                    continue

                child.depth = node.depth + 1
                self.visited[child] = node

                # goal node found
                if child == end:
                    return self._backtrack(start, end)

                # extend search
                self.frontier.put(child)

            # limit number of nodes explored
            if max_iters < 1:
                break

        # no goal found (within maximum iterations)
        return []

    def show(self, start, end, max_iters=10000) -> None:
        """TODO"""


# ------------------------------------------------- Iterative Deepening Search
class IterativeDeepening(Search):
    """Iterative Deepening Search Algorithm"""

    def __init__(self, max_depth: int=MAX_INT) -> None:
        """Instance of the iterative deepening search algorithm for
        pathfinding in node based environments

        Args:
            ``max_depth``: Optional depth limit to do Depth Limited Search
        """

        super().__init__(queue_type=LifoQueue())
        self.max_depth = max_depth

    def find(self,
        start: Node,
        end: Node,
        max_iters: int=10000
    ) -> List[Node]:
        """Use the iterative deepening search algorithm to find the path from
        a start node to the goal node, limited by a maximum number of nodes to
        check

        Args:
            ``start``: Initial node to start the search from
            
            ``end``: Goal node to be searched for
            
            ``max_iters``: Maximum number of nodes to be visited
                (default: 10000)

        Returns:
            list: path to goal node or an empty list if goal was not reached
        """

        for depth in range(self.max_depth):
            # perform depth first search at each max depth level
            dfs = DepthFirst(max_depth=depth)
            path = dfs.find(start, end, max_iters)
            max_iters -= dfs.iterations

            if path:
                return path

        # no goal found (within maximum iterations)
        return []

    def show(self, start, end, max_iters=10000) -> None:
        """TODO"""


# -------------------------------------------------------------- A Star Search
class AStar(Search):
    """A Star Search Algorithm"""

    def __init__(self) -> None:
        """Instance of the A star search algorithm for pathfinding in node
        based environments
        """

        super().__init__(queue_type=PriorityQueue())

    def find(self,
        start: Node,
        end: Node,
        max_iters: int=10000
    ) -> List[Node]:
        """Use the A star search algorithm to find the path from a start node
        to the goal node, prioritized by path cost and heuristic value and
        limited by a maximum number of nodes to check

        Args:
            ``start``: Initial node to start the search from
            
            ``end``: Goal node to be searched for
            
            ``max_iters``: Maximum number of nodes to be visited
                (default: 10000)

        Returns:
            list: path to goal node or an empty list if goal was not reached
        """

        start.info['priority'] = start.cost + start.heuristic
        self.frontier.put(start)
        self.visited[start] = None

        # start node is goal node
        if start == end:
            return [start]

        # limit loops and stop if no more nodes available
        while not self.frontier.empty():
            max_iters -= 1
            node = self.frontier.get()

            # check neighbors
            for child in node.neighbors:
                # avoid infinite loops
                if child in self.visited:
                    continue

                self.visited[child] = node

                # goal node found
                if child == end:
                    return self._backtrack(start, end)

                # extend search
                child.info['priority'] = child.cost + child.heuristic
                self.frontier.put(child)

            # limit number of nodes explored
            if max_iters < 1:
                break

        # no goal found (within maximum iterations)
        return []

    def show(self, start, end, max_iters=10000) -> None:
        """TODO"""
