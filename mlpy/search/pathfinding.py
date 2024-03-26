"""Pathfinding TODO"""

from typing import List

from queue import Queue, PriorityQueue
from mlpy.types import Node, Search


# ------------------------------------------------------- Breadth First Search
class BreadthFirst(Search):
    def __init__(self) -> None:
        """Instance of the breadth first search algorithm for pathfinding in
        node based environments.
        """

        super().__init__(queue_type=Queue())
    
    def find(self,
        start: Node,
        end: Node,
        max_iters: int=10000
    ) -> List[Node]:
        """Use the breadth first search algorithm to find the path from a
        start node to the goal node, limited by a maximum number of nodes to
        check.

        Args:
            ``start``: Initial node to start the search from
            
            ``end``: Goal node to be searched for
            
            ``max_iters``: Maximum number of nodes to be visited
                (default: 10000)

        Returns:
            list: path to goal node or an empty list if goal was not reached
        """
        
        self.path.append(start)
        self.frontier.put(start)
        self.visited.append(start)
        
        # start node is goal node
        if start == end:
            return self.path
        
        # limit loops and stop if no more nodes available
        while max_iters > 0 and not self.frontier.empty():
            max_iters -= 1
            node = self.frontier.get()
            self.visited.append(node)

            # check neighbors
            for child in node.neighbors:
                # avoid infinite loops
                if child in self.visited:
                    continue

                # TODO path needs to get correct intermediate nodes
                # use child.data["path"] = ...

                # goal node found
                if child == end:
                    self.path.append(child)
                    return self.path
                
                # extend search
                self.frontier.put(child)

        # no goal found (within maximum iterations)
        self.path = []
        return self.path
    

# --------------------------------------------------- Greedy Best First Search
class GreedyBestFirst(Search):
    def __init__(self) -> None:
        """Instance of the greedy best first search algorithm for pathfinding
        in node based environments.
        """

        super().__init__(queue_type=PriorityQueue())
    
    def find(self,
        start: Node,
        end: Node,
        max_iters: int=10000
    ) -> List[Node]:
        """Use the greedy best first search algorithm to find the path from a
        start node to the goal node, limited by a maximum number of nodes to
        check.

        Args:
            ``start``: Initial node to start the search from
            
            ``end``: Goal node to be searched for
            
            ``max_iters``: Maximum number of nodes to be visited
                (default: 10000)

        Returns:
            list: path to goal node or an empty list if goal was not reached
        """

        self.path.append(start)
        self.frontier.put(0, start)
        self.visited.append(start)

        # start node is goal node
        if start == end:
            return self.path
        
        # limit loops and stop if no more nodes available
        while max_iters > 0 and not self.frontier.empty():
            max_iters -= 1
            node = self.frontier.get()
            self.visited.append(node)

            # TODO

        # no goal found (within maximum iterations)
        self.path = []
        return self.path
    
    
# --------------------------------------------------------- Depth First Search
class DepthFirst(Search):
    def __init__(self) -> None:
        """Instance of the depth first search algorithm for pathfinding in
        node based environments.
        """

        super().__init__(queue_type=Queue())
    
    def find(self,
        start: Node,
        end: Node,
        max_iters: int=10000
    ) -> List[Node]:
        """Use the depth first search algorithm to find the path from a start
        node to the goal node, limited by a maximum number of nodes to check.

        Args:
            ``start``: Initial node to start the search from
            
            ``end``: Goal node to be searched for
            
            ``max_iters``: Maximum number of nodes to be visited
                (default: 10000)

        Returns:
            list: path to goal node or an empty list if goal was not reached
        """

        # TODO

        return super().find(start, end, max_iters)


# ------------------------------------------------- Iterative Deepening Search
class IterativeDeepening(Search):
    def __init__(self) -> None:
        """Instance of the iterative deepening search algorithm for
        pathfinding in node based environments.
        """

        super().__init__(queue_type=Queue())
    
    def find(self,
        start: Node,
        end: Node,
        max_iters: int=10000
    ) -> List[Node]:
        """Use the iterative deepening search algorithm to find the path from
        a start node to the goal node, limited by a maximum number of nodes to
        check.

        Args:
            ``start``: Initial node to start the search from
            
            ``end``: Goal node to be searched for
            
            ``max_iters``: Maximum number of nodes to be visited
                (default: 10000)

        Returns:
            list: path to goal node or an empty list if goal was not reached
        """

        # TODO

        return super().find(start, end, max_iters)
    

# -------------------------------------------------------------- A Star Search
class AStar(Search):
    def __init__(self) -> None:
        """Instance of the A star search algorithm for pathfinding in node
        based environments.
        """

        super().__init__(queue_type=Queue())
    
    def find(self,
        start: Node,
        end: Node,
        max_iters: int=10000
    ) -> List[Node]:
        """Use the A star search algorithm to find the path from a start node
        to the goal node, limited by a maximum number of nodes to check.

        Args:
            ``start``: Initial node to start the search from
            
            ``end``: Goal node to be searched for
            
            ``max_iters``: Maximum number of nodes to be visited
                (default: 10000)

        Returns:
            list: path to goal node or an empty list if goal was not reached
        """

        # TODO

        return super().find(start, end, max_iters)
