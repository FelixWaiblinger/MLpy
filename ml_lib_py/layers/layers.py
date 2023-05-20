"""
Module containing a range of layers commonly used in neural networks and
    similar applications.
"""


# Fully-connected layer ------------------------------------------------------
class Dense():
    def __init__(self, units, a_func):
        self.num_units = units
        self.a_func = a_func
