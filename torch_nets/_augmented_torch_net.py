
__module_name__ = "__init__.py"
__doc__ = """
          Module contianing the API for accessing AugmentedTorchNets. This module
          remains under active development and testing.
          """
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])
__version__ = "0.0.2"


# -- import packages: --------------------------------------------------------------------
from typing import Union
import torch


# -- import local dependencies: ----------------------------------------------------------
from ._torch_net import _torch_net as TorchNet


# -- primary module: ---------------------------------------------------------------------
class AugmentedTorchNet(torch.nn.Module):
    """
    TorchNet with an additional torch.nn.Linear layer to transform in_dim + n_aug -> out_dim
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
        n_augment: int = 0,
        bias: bool = True,
        output_bias: bool = True,
    ):
        """
        # TODO: docs.
        
        Augmented TorchNet.
        
        Parameters:
        -----------
        
        Returns:
        --------
        None
        
        """
        super(AugmentedTorchNet, self).__init__()

        self.__parse__(locals())
        self.net

    def __parse__(self, kwargs, ignore=["self"]):
        for key, val in kwargs.items():
            if not key in ignore:
                setattr(self, key, val)

    def _configure_neural_net(self):
        self.torch_net = TorchNet(
            self.in_features,
            self.out_features,
            self.hidden,
            self.activation,
            self.dropout,
            self.n_augment,
            self.bias,
            self.output_bias,
        )

    @property
    def net(self):
        if not hasattr(self, "torch_net"):
            self._configure_neural_net()
        return self.torch_net

    @property
    def augmented_output_layer(self):
        aug_dim = int(self.out_features + self.n_augment)
        return torch.nn.Linear(aug_dim, self.out_features)

    def augmented_input(self, input):
        x_aug = torch.zeros(input.shape[0], self.n_augment)
        return torch.cat([input, x_aug], 1)

    def forward(self, input):
        return self.augmented_output_layer(self.net(self.augmented_input(input)))
