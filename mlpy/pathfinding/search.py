"""Search TODO"""

from queue import Queue, PriorityQueue
from mlpy.types import Node, Search


# ------------------------------------------------------- Breadth First Search
class BreadthFirst(Search):
    def __init__(self) -> None:
        super().__init__(queue_type=Queue())
    
    def find(self, start, end, max_iters=10000):
        """TODO
        """

        # TODO

        return super().find(start, end, max_iters)
    

# --------------------------------------------------- Greedy Best First Search
class GreedyBestFirst(Search):
    def __init__(self) -> None:
        super().__init__(queue_type=Queue())
    
    def find(self, start, end, max_iters=10000):
        """TODO
        """

        # TODO

        return super().find(start, end, max_iters)
    
    
# --------------------------------------------------------- Depth First Search
class DepthFirst(Search):
    def __init__(self) -> None:
        super().__init__(queue_type=Queue())
    
    def find(self, start, end, max_iters=10000):
        """TODO
        """

        # TODO

        return super().find(start, end, max_iters)


# ------------------------------------------------- Iterative Deepening Search
class IterativeDeepening(Search):
    def __init__(self) -> None:
        super().__init__(queue_type=Queue())
    
    def find(self, start, end, max_iters=10000):
        """TODO
        """

        # TODO

        return super().find(start, end, max_iters)
    

# -------------------------------------------------------------- A Star Search
class AStar(Search):
    def __init__(self) -> None:
        super().__init__(queue_type=Queue())
    
    def find(self, start, end, max_iters=10000):
        """TODO
        """

        # TODO

        return super().find(start, end, max_iters)
