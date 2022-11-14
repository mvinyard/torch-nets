
__module_name__ = "_torch_net.py"
__doc__ = """Main user-facing API for torch.nn.Sequential."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu"])


# -- import packages:  ------------------------------------------------------
from collections import OrderedDict
from itertools import groupby
from typing import Union, Any
import torch


# -- import local dependencies: ---------------------------------------------
from .core._layer import Layer


# -- Supporting functions: --------------------------------------------------
def as_list(input: Union[list, Any]):
    """Convert to list, if not already"""
    if isinstance(input, list):
        return input
    return [input]


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


def is_uniform(iterable):
    """Evaluate if all items in a list are uniform"""
    grouped = groupby(iterable)
    return next(grouped, True) and not next(grouped, False)


def format_layer_args(layer_arg, n_hidden):
    """format layer arguments w.r.t. the # of hidden layers"""
    layer_args = as_list(layer_arg)

    if is_uniform(layer_args):
        return [layer_args[0]] * n_hidden

    if len(layer_args) != n_hidden:
        n_missing = n_hidden - len(layer_args)
        return layer_args + [layer_args[-1]] * n_missing

    else:
        return layer_args


# -- Main class: ------------------------------------------------------------
def TorchNet(
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
    TorchNetStructure = define_structure(in_features, out_features, hidden)

    kwargs = {}
    for key in ["activation", "bias", "dropout"]:
        kwargs[key] = format_layer_args(locals()[key], n_hidden=len(as_list(hidden)))

    TorchNetDict = OrderedDict()
    for n, (name, layer_dims) in enumerate(TorchNetStructure.items()):
        if name != "output":
            TorchNetDict[name] = Layer(
                in_features=layer_dims[0],
                out_features=layer_dims[1],
                activation=kwargs["activation"][n],
                bias=kwargs["bias"][n],
                dropout=kwargs["dropout"][n],
            )()
        else:
            TorchNetDict[name] = Layer(
                in_features=layer_dims[0],
                out_features=layer_dims[1],
                bias=output_bias,
            )()

    return torch.nn.Sequential(TorchNetDict)
