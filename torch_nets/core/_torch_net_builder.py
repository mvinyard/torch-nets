
# -- import packages: ----------------------------------------------------------
from typing import List, Union, Any
from collections import OrderedDict
from ABCParse import ABCParse
import torch


# -- import local dependencies: ------------------------------------------------
from ._network_structure_config import NetworkStructureConfig
from ._torch_net_args_config import TorchNetArgsConfig
from ._layer_builder import LayerBuilder


# -- main class: ---------------------------------------------------------------
class TorchNetBuilder(ABCParse):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden=List[int],
        activation: str = ["LeakyReLU"],
        dropout: float = [0.2],
        bias: bool = [True],
        output_bias: bool = True,
    ):
        self.__parse__(kwargs=locals())

    def _as_list(self, input: Union[list, Any]):
        """Convert to list, if not already"""
        if isinstance(input, list):
            return input
        return [input]

    @property
    def _building_list(self):
        return ["hidden", "activation", "bias", "dropout"]

    @property
    def n_hidden(self):
        return len(self._as_list(self._PARAMS["hidden"]))

    @property
    def architecture(self):
        network_structure = NetworkStructureConfig()
        return network_structure(self.in_features, self.out_features, self.hidden)

    def stack(self):
        args_config = TorchNetArgsConfig(self.n_hidden)

        for key, val in self._PARAMS.items():
            if key in self._building_list:
                val = args_config(self._PARAMS[key])
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

        for n, (layer_name, (in_dim, out_dim)) in enumerate(self.architecture.items()):
            if layer_name == "output":
                TorchNetDict[layer_name] = self._build_output_layer(in_dim, out_dim)
            else:
                TorchNetDict[layer_name] = self._build_hidden_layer(in_dim, out_dim, n)

        return TorchNetDict

    def __call__(self):
        return torch.nn.Sequential(OrderedDict(self.__build__()))
