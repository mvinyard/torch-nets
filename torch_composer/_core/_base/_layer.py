
__module_name__ = "_layer.py"
__doc__ = """Layer module."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu"])


# -- import packages: --------------------------------------------------------------------
from abc import ABC
from collections import OrderedDict
import torch


# -- Base Layer module: ------------------------------------------------------------------
class Layer(ABC):
    __name__ = ""

    def __init__(
        self,
        in_features,
        out_features,
        activation=None,
        bias=True,
        dropout=0,
        name: str = None,
    ):
        """
        
        Parameters:
        -----------
        in_features
        
        out_features
        
        activation
        
        bias
        
        dropout
        
        name
        
        Returns:
        --------
        
        
        Notes:
        ------
        (1) General flow assumed is: Linear -> Dropout -> Activation
        """
        
        self.__parse_kwargs__(name, locals())

    # --- utilities: ---------------------------------------------------------------------
    def __configure_activation__(self, func):

        if isinstance(func, str):
            return getattr(torch.nn, func)()
        if isinstance(func, torch.nn.Module):
            return func
        if isinstance(func(), torch.nn.Module):
            return func()

        print(
            "Must pass torch.nn.<function> or a string that fetches a torch.nn.<function>"
        )

    def __parse_kwargs__(self, name, kwargs, ignore=["self", "name"]):

        setattr(self, "__name__", name)

        self._kwargs = {}
        for k, v in kwargs.items():
            if k and (not k in ignore):
                self._kwargs[k] = v
                setattr(self, "_{}".format(k), v)

    # -- core properties: ----------------------------------------------------------------
    @property
    def linear(self):
        return torch.nn.Linear(
            in_features=self._in_features,
            out_features=self._out_features,
            bias=self._bias,
        )

    @property
    def dropout(self):
        if self._dropout:
            return torch.nn.Dropout(self._dropout)

    @property
    def activation(self):
        if self._activation:
            return self.__configure_activation__(self._activation)

    # -- called: -------------------------------------------------------------------------
    def __collect_attributes__(self):

        attributes = [i for i in layer.__dir__() if not i.startswith("_")]
        for attr in attributes:
            if not getattr(self, attr) is None:
                if self.__name__:
                    attr_name = "_".join([self.__name__, attr])
                else:
                    attr_name = attr
                yield (attr_name, getattr(self, attr))

    def __call__(self) -> dict:
        return torch.nn.Sequential(OrderedDict(self.__collect_attributes__()))
