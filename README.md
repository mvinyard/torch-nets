# Torch-Nets

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/torch-composer.svg)](https://pypi.python.org/pypi/torch-composer/)
[![PyPI version](https://badge.fury.io/py/torch-composer.svg)](https://badge.fury.io/py/torch-composer)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


<a href="https://github.com/mvinyard/torch-nets/"><img src="/docs/imgs/ol-reliable-spongebob.gif" alt="ol-reliable-spongebob" width="400"/></a>

Compose PyTorch neural networks with ease.

#### Installation

From PYPI (current version: [`v0.0.1`](https://pypi.org/project/torch-nets/0.0.1/))
```python
pip install torch-nets
```

Alternatively, install the development version from GitHub:
```shell
git clone https://github.com/mvinyard/torch-nets.git;
cd torch-nets; pip install -e .
```

### Example API use-case

```python
from torch_nets import TorchNet
```

#### Create a feed-forward neural network

```python
net = TorchNet(
    in_features=50,
    out_features=50,
    hidden=[400, 400],
    activation="LeakyReLU",
    dropout=0.2,
    bias=True,
    output_bias=True,
)
net
```
```
Sequential(
  (hidden_1): Sequential(
    (linear): Linear(in_features=50, out_features=400, bias=True)
    (dropout): Dropout(p=0.2, inplace=False)
    (activation): LeakyReLU(negative_slope=0.01)
  )
  (hidden_2): Sequential(
    (linear): Linear(in_features=400, out_features=400, bias=True)
    (dropout): Dropout(p=0.2, inplace=False)
    (activation): LeakyReLU(negative_slope=0.01)
  )
  (output): Sequential(
    (linear): Linear(in_features=400, out_features=50, bias=True)
  )
)
```

The only required arguments are `in_features` and `out_features`. The network can be made as simple or complex as you want through optional parameters.


#### Potential future plans

- Composition of `torch.optim` funcs.
