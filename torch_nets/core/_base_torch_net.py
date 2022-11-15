
# -- import packages: --------------------------------------------------------------------
from abc import ABC, abstractmethod
import torch


# -- import local dependencies: ----------------------------------------------------------
from ._support_functions import as_list, is_uniform


# -- supporting functions: ---------------------------------------------------------------
def format_layer_args(layer_arg, n_hidden):
    """format layer arguments w.r.t. the # of hidden layers"""
    layer_args = as_list(layer_arg)
        
    if len(layer_args) == 0:
        return layer_args

    if is_uniform(layer_args):
        return [layer_args[0]] * n_hidden

    if len(layer_args) != n_hidden:
        n_missing = n_hidden - len(layer_args)
        return layer_args + [layer_args[-1]] * n_missing

    else:
        return layer_args


# -- Base class: -------------------------------------------------------------------------
class BaseTorchNet(ABC):
    def __init__(
        self,
        in_features,
        out_features,
        hidden,
        activation="LeakyReLU",
        dropout=0.2,
        bias=True,
        output_bias=True,
    ):

        self.__parse__(locals())

    def __parse__(
        self,
        kwargs: dict,
        mk_list: list = ["hidden", "activation", "bias", "dropout"],
        ignore=["self"],
    ):

        ## should have one of these two scenarios
        if "hidden" in kwargs.keys():
            self.n_hidden = len(as_list(kwargs["hidden"]))
        else:
            self.n_hidden = kwargs["n_hidden"]
        ## --------------------------------------

        for key, val in kwargs.items():
            if not key in ignore:
                if key in mk_list:
                    val = format_layer_args(kwargs[key], n_hidden=self.n_hidden)
                setattr(self, key, val)

    @abstractmethod
    def __build__(self):
        """Must return OrderedDict"""

    def __call__(self):
        return torch.nn.Sequential(self.__build__())