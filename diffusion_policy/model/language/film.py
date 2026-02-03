import torch
import torch.nn as nn


class FiMLayer(nn.Module):
    """
    Simple FiLM layer that modulates features x with conditioning vector cond.

    x: (B, To, D_feat)
    cond:
        - (B, D_cond): broadcast over To
        - (B, To, D_cond): per-timestep conditioning
    """

    def __init__(self, feature_dim: int, cond_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        self.cond_dim = cond_dim
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must be 3D (B, To, D_feat), got {x.shape}")

        B, To, D_feat = x.shape
        if D_feat != self.feature_dim:
            raise ValueError(
                f"Feature dim mismatch: expected {self.feature_dim}, got {D_feat}"
            )

        if cond.dim() == 2:
            # (B, D_cond) -> (B, To, D_cond)
            cond = cond.unsqueeze(1).expand(-1, To, -1)
        elif cond.dim() == 3:
            if cond.shape[1] != To:
                raise ValueError(
                    f"Time dim mismatch between x and cond: {To} vs {cond.shape[1]}"
                )
        else:
            raise ValueError(
                f"cond must have shape (B, D_cond) or (B, To, D_cond), got {cond.shape}"
            )

        cond_flat = cond.reshape(B * To, -1)
        gamma_beta = self.mlp(cond_flat)  # (B*To, 2*D_feat)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gamma = gamma.view(B, To, D_feat)
        beta = beta.view(B, To, D_feat)

        return x * (1.0 + gamma) + beta

