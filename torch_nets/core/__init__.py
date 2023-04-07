
__module_name__ = "__init__.py"
__doc__ = """ __init__.py module for the API core."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])


# -- import network modules: -------------------------------------------------------------
from ._layer import Layer


from ._layer_args_config import LayerArgsConfig
from ._activation_function_config import ActivationFunctionConfig
from ._network_structure_config import NetworkStructureConfig