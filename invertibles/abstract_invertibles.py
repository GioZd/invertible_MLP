from typing import Protocol, runtime_checkable

import torch

@runtime_checkable
class InvertibleLayer(Protocol):
    """A structural type for any layer that supports inversion."""
    def inverse(self, y: torch.Tensor) -> torch.Tensor: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...