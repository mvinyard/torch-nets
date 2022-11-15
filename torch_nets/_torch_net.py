
__module_name__ = "_torch_net.py"
__doc__ = """Main user-facing API for torch.nn.Sequential."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu"])
__version__ = "0.0.1"


# -- import packages: --------------------------------------------------------------------
from collections import OrderedDict
from itertools import groupby
from typing import Union, Any
import torch


# -- import local dependencies: ----------------------------------------------------------
from .core._layer import Layer
from .core._support_functions import define_structure
from .core._base_torch_net import BaseTorchNet


# -- Main module class: ------------------------------------------------------------------
class TorchNet(BaseTorchNet):
    def __build__(self):

        TorchNetStructure = define_structure(
            self.in_features, self.out_features, self.hidden
        )

        TorchNetDict = OrderedDict()
        for n, (name, layer_dims) in enumerate(TorchNetStructure.items()):
            if name != "output":
                TorchNetDict[name] = Layer(
                    in_features=layer_dims[0],
                    out_features=layer_dims[1],
                    activation=self.activation[n],
                    bias=self.bias[n],
                    dropout=self.dropout[n],
                )()
            else:
                TorchNetDict[name] = Layer(
                    in_features=layer_dims[0],
                    out_features=layer_dims[1],
                    bias=self.output_bias,
                )()

        return TorchNetDict
    


# -- Main API-facing function: ----------------------------------------------
def _torch_net(
    in_features: int,
    out_features: int,
    hidden: Union[list, int] = [],
    activation="LeakyReLU",
    dropout: Union[float, list] = 0.2,
    bias: bool = True,
    output_bias: bool = True,
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

    hidden
        list of hidden layer sizes
        type: Union[list, int]

    activation
        If passed, defines appended activation function.
        type: 'torch.nn.modules.activation.<func>'
        default: None

    dropout
        If > 0, append dropout layer with probablity p, where p = dropout.
        type: float
        default: 0

    bias
        Indicate if the layer should/should not learn an additive bias.
        type: bool
        default: True

    output_bias
        Indicate if the output layer should/should not learn an
        additive bias.
        type: bool
        defualt: True


    Returns:
    --------
    TorchNet
        Neural network block. To be accepted into a torch.nn.Module object.
        type: torch.nn.Sequential

    Notes:
    ------
    (1) For params: 'activation', 'bias', and 'dropout': if more
        params than necessary are passed, they will go unused.
    """

    return TorchNet(
        in_features=in_features,
        out_features=out_features,
        hidden=hidden,
        activation=activation,
        dropout=dropout,
        bias=bias,
        output_bias=output_bias,
    )()
