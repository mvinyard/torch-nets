
__module_name__ = "_layer.py"
__doc__ = """Layer module."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu"])
__version__ = "0.0.1"

# -- import packages: --------------------------------------------------------------------
from abc import ABC
from collections import OrderedDict
import torch


# -- Base Layer module: ------------------------------------------------------------------
class Layer(ABC):
    __name__ = ""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: torch.nn.modules.activation = None,
        bias: bool = True,
        dropout: float = 0,
        name: str = None,
    ):
        """
        
        Parameters:
        -----------
        in_features
            Size of layer input.
            type: int
        
        out_features
            Size of layer output.
            type: int
        
        activation
            If passed, defines appended activation function.
            type: 'torch.nn.modules.activation.<func>'
            default: None
        
        bias
            Indicate if the layer should not learn an additive bias.
            type: bool
            default: True
            
        dropout
            If > 0, append dropout layer with probablity p, where p = dropout.
            type: float
            default: 0
        
        name [Optional]
            Name of the layer. If defined, specified within torch.nn.Sequential.
            Layer name.
            type: str
                    
        Returns:
        --------
        None, instantiates Layer class.
        
        Notes:
        ------
        (1) General flow assumed is: Linear -> Dropout -> Activation
        """
        
        self.__parse_kwargs__(name, locals())

    # --- utilities: ---------------------------------------------------------------------
    def __configure_activation__(self, func):
        """Register the passed activation function."""
        if isinstance(func, str):
            return getattr(torch.nn, func)()
        if isinstance(func, torch.nn.Module):
            return func
        if isinstance(func(), torch.nn.Module):
            return func()
        
        msg = "Must pass torch.nn.<function> or a string that fetches a torch.nn.<function>"
        print(msg)

    def __parse_kwargs__(self, name, kwargs, ignore=["self", "name"]):
        """Register passed keyword arguments. Called on self.__init__()"""
        setattr(self, "__name__", name)

        self._kwargs = {}
        for k, v in kwargs.items():
            if k and (not k in ignore):
                self._kwargs[k] = v
                setattr(self, "_{}".format(k), v)

    # -- core properties: ----------------------------------------------------------------
    @property
    def linear(self):
        """torch.nn.Linear layer"""
        return torch.nn.Linear(
            in_features=self._in_features,
            out_features=self._out_features,
            bias=self._bias,
        )

    @property
    def dropout(self):
        """torch.nn.Dropout layer."""
        if self._dropout:
            return torch.nn.Dropout(self._dropout)

    @property
    def activation(self):
        """torch.nn.<activation> layer"""
        if self._activation:
            return self.__configure_activation__(self._activation)

    # -- called: -------------------------------------------------------------------------
    def __collect_attributes__(self):
        """Collect passed layer and optionally dropout, activation."""
        attributes = [i for i in self.__dir__() if not i.startswith("_")]
        for attr in attributes:
            if not getattr(self, attr) is None:
                if self.__name__:
                    attr_name = "_".join([self.__name__, attr])
                else:
                    attr_name = attr
                yield (attr_name, getattr(self, attr))

    def __call__(self)-> torch.nn.Sequential:
        """
        Generate layer from arguments passed to __init__() and processed with supporting
        functions.
        
        Returns:
        --------
        Layer
            Composed layer
            type: torch.nn.Sequential
        """
        return torch.nn.Sequential(OrderedDict(self.__collect_attributes__()))
