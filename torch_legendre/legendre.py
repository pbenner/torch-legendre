import torch
from torch import nn


class LegendreLayer(nn.Module):
    """
    A neural network layer that expands each input feature into its Legendre polynomial
    basis up to a specified degree, then optionally applies a trainable Linear mapping
    to produce a desired output size.

    Parameters
    ----------
    in_features : int
        Number of input features.
    degree : int
        Number of polynomial degrees to compute per input feature.
        degree = 1 means only the constant term P0(x) = 1.
    out_features : int or None, optional
        If provided, the Legendre basis is followed by a linear layer mapping
        from (in_features * degree) → out_features.
        If None, the raw Legendre-expanded features are returned.
    bias : bool, default=True
        Whether to include a bias term in the optional linear mapping.
    """

    def __init__(
        self, in_features: int, degree: int, out_features: int = None, bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.degree = degree

        if out_features is not None:
            self.out_features = out_features
        else:
            self.out_features = in_features * degree

        # Precompute recurrence coefficients for Legendre polynomials
        if degree > 1:
            n = torch.arange(2, degree, dtype=torch.float32)
            self.register_buffer('coef_a', (2 * n - 1) / n)
            self.register_buffer('coef_b', (n - 1) / n)

        # Optional final linear mapping
        if out_features is not None:
            self.linear = nn.Linear(in_features * degree, out_features, bias=bias)
        else:
            self.linear = None

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, in_features)

        Returns
        -------
        torch.Tensor
            If out_features is None:
                Shape (batch_size, in_features * degree)
            Else:
                Shape (batch_size, out_features)
        """
        batch_size = x.shape[0]

        # Generate polynomial terms for each feature independently
        polys_list = [
            torch.ones(batch_size, self.in_features, device=x.device, dtype=x.dtype)
        ]
        if self.degree > 1:
            polys_list.append(x)
            for a, b in zip(self.coef_a, self.coef_b):
                polys_list.append(a * x * polys_list[-1] - b * polys_list[-2])

        # Stack into shape (batch, in_features, degree)
        polys = torch.stack(polys_list, dim=2)

        # Flatten degrees into features
        poly_feats = polys.reshape(batch_size, self.in_features * self.degree)

        # Optional trainable projection
        if self.linear is not None:
            return self.linear(poly_feats)

        return poly_feats
