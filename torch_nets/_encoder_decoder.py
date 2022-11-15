
__module_name__ = "_encoder_decoder.py"
__doc__ = """Main user-facing API for torch.nn.Sequential encoders/decoders."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu"])
__version__ = "0.0.1"


# -- import packages: --------------------------------------------------------------------
from typing import Union, Any
from collections import OrderedDict


# -- import local dependencies: ----------------------------------------------------------
from .core._base_torch_net import BaseTorchNet
from .core._support_functions import power_space, define_structure
from .core._layer import Layer


# -- Main module class: ------------------------------------------------------------------
class EncoderDecoder(BaseTorchNet):
    def __init__(
        self,
        in_features,
        out_features,
        n_hidden=3,
        power=2,
        activation="LeakyReLU",
        dropout=0.2,
        bias=True,
        output_bias=True,
    ):

        self.__parse__(locals())

    def __build__(self):

        structure = power_space(
            start=self.in_features,
            stop=self.out_features,
            n=(self.n_hidden + 2),
            power=self.power,
        ).tolist()

        TorchNetStructure = define_structure(
            in_features=structure[0], out_features=structure[-1], hidden=structure[1:-1]
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


# -- Main API-facing functions: ----------------------------------------------------------
def Encoder(
    in_features: int,
    out_features: int,
    n_hidden: int,
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

    n_hidden
        Number of hidden layers
        type: int

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

    return EncoderDecoder(
        in_features=in_features,
        out_features=out_features,
        n_hidden=n_hidden,
        activation=activation,
        dropout=dropout,
        bias=bias,
        output_bias=output_bias,
    )()


def Decoder(
    in_features: int,
    out_features: int,
    n_hidden: int,
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

    n_hidden
        Number of hidden layers
        type: int

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

    return EncoderDecoder(
        in_features=in_features,
        out_features=out_features,
        n_hidden=n_hidden,
        activation=activation,
        dropout=dropout,
        bias=bias,
        output_bias=output_bias,
    )()
