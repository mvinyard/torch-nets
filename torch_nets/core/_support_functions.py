
__module_name__ = "_support_functions.py"
__doc__ = """Support module."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])
__version__ = "0.0.1"


# -- import packages: --------------------------------------------------------------------
from itertools import groupby
from typing import Union, Any
import numpy as np


# -- Supporting functions: ---------------------------------------------------------------
def as_list(input: Union[list, Any]):
    """Convert to list, if not already"""
    if isinstance(input, list):
        return input
    return [input]


def is_uniform(iterable):
    """Evaluate if all items in a list are uniform"""
    grouped = groupby(iterable)
    return next(grouped, True) and not next(grouped, False)


def power_space(start: int, stop: int, n: int, power: Union[int, float]):
    """
    Return integered-powered space.

    Parameters:
    -----------
    start
        first term of the power-space array.
        type: int

    stop
        final term of the power-space array.
        type: int

    n
        type: int
        length of array to be created

    power
        type: Union[int, float]
        power at which the space should decay / expand.

    Returns:
    --------
    power-spaced array
        type: np.ndarray
    """
    start_ = np.power(start, 1 / float(power))
    stop_  = np.power(stop, 1 / float(power))

    pspace = np.power(np.linspace(start_, stop_, num=n), power).astype(int)
    pspace[0], pspace[-1] = start, stop
    
    return pspace


def define_structure(in_features: int, out_features: int, hidden: Union[int, list]):
    
    """Build layered neural network structure"""
    hidden = as_list(hidden)
    n_hidden = len(hidden)

    layer_names = ["hidden_{}".format(i + 1) for i in range(n_hidden)] + ["output"]
    structure = [in_features] + hidden + [out_features]

    TorchNetDict = {}

    for n, (i, j) in enumerate(zip(structure[:-1], structure[1:])):
        TorchNetDict[layer_names[n]] = (i, j)

    return TorchNetDict
