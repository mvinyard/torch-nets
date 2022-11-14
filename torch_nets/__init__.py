
__module_name__ = "__init__.py"
__doc__ = """Main API __init__.py module."""
__author__ = ", ".join(["Michael E. Vinyard"])
__email__ = ", ".join(["vinyard@g.harvard.edu",])
__version__ = "0.0.1rc"


# -- import network modules: -------------------------------------------------------------
from ._torch_net import TorchNet


# -- import API core: --------------------------------------------------------------------
from . import core
