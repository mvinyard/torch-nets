
__module_name__ = "__init__.py"
__doc__ = """
          Module contianing the API for accessing AugmentedTorchNets. This module
          remains under active development and testing.
          """
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# -- import packages: ----------------------------------------------------------
import torch
from typing import Union
from ABCParse import ABCParse


# -- import local dependencies: ------------------------------------------------
from ._torch_net import TorchNet


# -- primary module: -----------------------------------------------------------
class AugmentedTorchNet(torch.nn.Module, ABCParse):
    """
    TorchNet with additional torch.nn.Linear layer.
    Transforms (in_dim + n_aug) -> out_dim.

    Source:
    -------
     - paper:  https://arxiv.org/abs/1904.01681
     - GitHub: https://github.com/EmilienDupont/augmented-neural-odes
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: Union[list, int] = [],
        activation="LeakyReLU",
        dropout: Union[float, list] = 0.2,
        bias: bool = True,
        n_augment: int = 0,
        output_bias: bool = True,
    ):
        """
        Augmented TorchNet.

        Parameters:
        -----------
        in_features
        
        out_features
        
        hidden
        
        activation
        
        dropout
        
        bias
        
        n_augment
        
        output_bias
        

        Returns:
        --------
        None, instantiates torch.nn.Module for augmented neural network.

        Examples:
        ---------
        
        """
        super(AugmentedTorchNet, self).__init__()

        self.__parse__(locals())
        self._configure_neural_net()

    def _configure_neural_net(self):
        self.torch_net = TorchNet(
            in_features=self.augmented_in,
            out_features=self.augmented_out,
            hidden=self.hidden,
            activation=self.activation,
            dropout=self.dropout,
            bias=self.bias,
            output_bias=self.output_bias,
        )

    @property
    def augmented_in(self):
        """updates self.in_features"""
        return self.in_features + self.n_augment

    @property
    def augmented_out(self):
        """updates self.out_features"""
        return self.out_features + self.n_augment

    @property
    def augmented_output_layer(self):
        return torch.nn.Linear(self.augmented_out, self.out_features)

    def augmented_input(self, input):
        x_aug = torch.zeros(
            input.shape[0], self.n_augment, device=input.device
        )
        return torch.cat([input, x_aug], 1)

    def forward(self, input):
        return self.augmented_output_layer(self.torch_net(self.augmented_input(input)))
