# ![logo](/docs/imgs/torch_composer.logo.large.svg)

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/torch-composer.svg)](https://pypi.python.org/pypi/torch-composer/)
[![PyPI version](https://badge.fury.io/py/torch-composer.svg)](https://badge.fury.io/py/torch-composer)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# torch-composer
Compose pytorch neural networks with ease.

### Installation (current version: [`v0.1.0`](https://pypi.org/project/torch-composer/))
```python
pip install torch-composer
```

### Sample use-cases of the API
```python
import torch_composer
import torch

net = torch_composer.TorchNet(in_dim=2500, out_dim=10)
```
```
Sequential(
  (input): Linear(in_features=2500, out_features=200, bias=True)
  (activation_1): LeakyReLU(negative_slope=0.01)
  (hidden_1): Linear(in_features=200, out_features=200, bias=True)
  (output_activation): LeakyReLU(negative_slope=0.01)
  (output): Linear(in_features=200, out_features=10, bias=True)
)
```

As simple as you want (see above) or more complex with optional parameters:
```python
torch_composer.TorchNet(
    in_dim=2500,
    out_dim=10,
    hidden={1: [800, 800], 2: [200, 200]},
    activation_function=torch.nn.LeakyReLU(negative_slope=0.01),
    dropout=0.2,
    input_bias=True,
    output_bias=True,
)
```
```
Sequential(
  (input): Linear(in_features=2500, out_features=800, bias=True)
  (activation_1): LeakyReLU(negative_slope=0.01)
  (hidden_1): Linear(in_features=800, out_features=800, bias=True)
  (dropout_1): Dropout(p=0.2, inplace=False)
  (activation_2): LeakyReLU(negative_slope=0.01)
  (hidden_2): Linear(in_features=200, out_features=200, bias=True)
  (dropout_2): Dropout(p=0.2, inplace=False)
  (output_activation): LeakyReLU(negative_slope=0.01)
  (output): Linear(in_features=200, out_features=10, bias=True)
)
```

### Make an encoder
```python=
torch_composer.TorchNetDecoder(data_dim=2500, latent_dim=10)
```

### Make a decoder
```python=
torch_composer.TorchNetDecoder(data_dim=2500, latent_dim=10)
```

Access and set initial parameters for the output layer:
```python=
torch_composer.tools.init_output_params(net)
```

#### Potential future plans

- Composition of `torch.optim` funcs.
