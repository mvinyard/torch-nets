
__module_name__ = "__init__.py"
__doc__ = """ __init__.py module for the API core."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# -- import network modules: -------------------------------------------------------------
from ._layer_builder import LayerBuilder


from ._torch_net_args_config import TorchNetArgsConfig
from ._activation_function_config import ActivationFunctionConfig
from ._network_structure_config import NetworkStructureConfig

from ._torch_net_builder import TorchNetBuilder