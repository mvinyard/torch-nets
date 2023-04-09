
from ABCParse import ABCParse
import torch
from typing import Union, List, Any
from collections import OrderedDict

from .core.config import Config
from .core import LayerBuilder

class TorchNet(torch.nn.Sequential, ABCParse):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: Union[List[int], int] = [],
        activation: Union[str, List[str]] = "LeakyReLU",
        dropout: Union[float, List[float]] = 0.2,
        bias: bool = True,
        output_bias: bool = True,
    ):
        self.__parse__(locals())

        self.config = Config(
            in_features=in_features,
            out_features=out_features,
            hidden=hidden,
        )

        self.layers = []
        self.lb = LayerBuilder()

        xyz = self.__build__()
        self.names = []
        for i, (x, y) in enumerate(xyz.items()):
            self.layers.append(y)
            self.names.append(x)

        super(TorchNet, self).__init__(*self.layers)
        self._rename_nn_sequential_inplace(self, self.names)

    def _rename_nn_sequential_inplace(
        self, sequential: torch.nn.Sequential, names: List[str]
    ) -> None:
        new_modules = OrderedDict()
        for i, (k, v) in enumerate(sequential._modules.items()):
            new_modules[names[i]] = v

        sequential._modules = new_modules

    @property
    def _building_list(self):
        return ["hidden", "activation", "bias", "dropout"]

    def stack(self):
        for key, val in self._PARAMS.items():
            if key in self._building_list:
                val = self.config.layerwise_attributes(self._PARAMS[key])
                setattr(self, key, val)

    def _build_hidden_layer(self, in_dim, out_dim, n):
        return LayerBuilder()(
            in_features=in_dim,
            out_features=out_dim,
            activation=self.activation[n],
            bias=self.bias[n],
            dropout=self.dropout[n],
        )

    def _build_output_layer(self, in_dim, out_dim):
        return LayerBuilder()(
            in_features=in_dim,
            out_features=out_dim,
            bias=self.output_bias,
        )

    def __build__(self):
        self.stack()

        TorchNetDict = {}

        for n, (layer_name, (in_dim, out_dim)) in enumerate(
            self.config.network_structure.items()
        ):
            if layer_name == "output":
                TorchNetDict[layer_name] = self._build_output_layer(in_dim, out_dim)
            else:
                TorchNetDict[layer_name] = self._build_hidden_layer(in_dim, out_dim, n)

        return TorchNetDict

    # No need to define forward() method as it is already handled by nn.Sequential
    def _as_list(self, input: Union[list, Any]):
        """Convert to list, if not already"""
        if isinstance(input, list):
            return input
        return [input]