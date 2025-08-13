# torch-legendre

A PyTorch layer for expanding input features into **Legendre polynomial bases**.
Useful for building models with polynomial feature expansions while keeping the
workflow compatible with standard PyTorch `nn.Module` layers.

## Features

- Computes **Legendre polynomial** terms P₀(x), P₁(x), …, Pₙ₋₁(x) for each input feature
- Supports arbitrary input dimensionality
- Optional trainable linear projection after expansion
- Drop-in compatible with `torch.nn.Sequential`
- Efficient recurrence-based computation (no loops over batches)

---

## Installation

```bash
pip install -e .
```

## Usage

You can see a full demonstration of the `LegendreLayer` in the provided [example.ipynb](example.ipynb) notebook.

### 1. Basic Legendre Expansion

```python
import torch
from torch_legendre import LegendreLayer

# Example: 2 input features, expand to degree 4 (P₀...P₃)
layer = LegendreLayer(in_features=2, degree=4)

x = torch.tensor([[0.1, -0.3],
                  [0.5,  0.2]])  # shape (batch=2, in_features=2)

y = layer(x)
print(y.shape)  # (2, 2 * 4) = (2, 8)
```

### 2. With Trainable Projection

```python
# Expand then project down to 3 outputs
layer = LegendreLayer(in_features=2, degree=4, out_features=3)

x = torch.rand(5, 2)
y = layer(x)
print(y.shape)  # (5, 3)
```

### 3. Stacking in a Sequential Model

```python
import torch.nn as nn

model = nn.Sequential(
    LegendreLayer(in_features=1, degree=5, out_features=10),
    LegendreLayer(in_features=10, degree=3, out_features=1)
)
```

## API

### `LegendreLayer`

Expands each input feature into its Legendre polynomial basis and optionally applies a trainable linear projection.

**Parameters**

- **in_features** (*int*):
  Number of input features.

- **degree** (*int*):
  Number of polynomial degrees to compute per input feature.
  `degree = 1` means only the constant term `P₀(x) = 1`.

- **out_features** (*int, optional*):
  If provided, a final `nn.Linear` layer maps from `(in_features * degree)` → `out_features`.
  If `None`, returns the raw expanded features.

- **bias** (*bool, default=True*):
  Whether to include a bias term in the optional linear mapping.

**Shapes**

- **Input**: `(batch_size, in_features)`
- **Output**:
  - If `out_features is None`: `(batch_size, in_features * degree)`
  - Else: `(batch_size, out_features)`

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
