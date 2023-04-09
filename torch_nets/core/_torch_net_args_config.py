
# -- import local dependenecies: -----------------------------------------------


# -- set typing: ---------------------------------------------------------------
from itertools import groupby
from typing import Any, Union
NoneType = type(None)


# -- operational class: --------------------------------------------------------
class TorchNetArgsConfig:
    """
    Example:
    --------
    >>> format_layer_args = FormatLayerArgs(n_hidden=3)
    >>> format_layer_args(layer_arg)

    """

    def __init__(self, n_hidden: int) -> NoneType:
        self.n_hidden = n_hidden

    def _as_list(self, input: Union[list, Any]):
        """Convert to list, if not already"""
        if isinstance(input, list):
            return input
        return [input]

    def _is_uniform(self, iterable):
        """Evaluate if all items in a list are uniform"""
        grouped = groupby(iterable)
        return next(grouped, True) and not next(grouped, False)

    @property
    def n_missing(self) -> int:
        return self.n_hidden - self.n_args

    def __call__(self, layer_arg: Any) -> Any:
        """format layer arguments w.r.t. the # of hidden layers"""

        layer_args = self._as_list(layer_arg)
        self.n_args = len(layer_args)

        if self.n_args == 0:
            return layer_args

        if self._is_uniform(layer_args):
            return [layer_args[0]] * self.n_hidden

        if self.n_args != self.n_hidden:
            return layer_args + [layer_args[-1]] * self.n_missing

        else:
            return layer_args
