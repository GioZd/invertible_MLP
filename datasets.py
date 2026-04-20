from typing import Tuple

import torch
import torch.nn.functional as F


def make_spiral(
    n_per_class: int = 500,
    noise: float = 0.1,
    turns: float = 1.5,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a 2-D two-class spiral (the classic “two-spiral” toy problem).

    Args
    ----
    n_per_class: number of points per class (total = 2 * n_per_class).
    noise: standard deviation of Gaussian noise added to each point.
    turns: how many revolutions each spiral makes (default 3).
    device: torch device for the returned tensors.

    Returns
    -------
    x: Tensor of shape (2*n_per_class, 2) - the coordinates.
    y: Tensor of shape (2*n_per_class, 2) - one-hot class labels (float).

    """
    # Angles
    theta = torch.linspace(0, 2 * turns * torch.pi, n_per_class, device=device)
    r = 1.2*theta + 1.0

    x0 = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    x1 = torch.stack([r * torch.cos(theta + torch.pi), r *
                     torch.sin(theta + torch.pi)], dim=1)

    x = torch.cat([x0, x1], dim=0)

    # Add Gaussian noise
    if noise > 0:
        x += torch.randn_like(x) * noise

    # Labels
    labels = torch.cat([
        torch.zeros(n_per_class, dtype=torch.long, device=device),
        torch.ones(n_per_class, dtype=torch.long, device=device)
    ], dim=0)

    y = F.one_hot(labels, num_classes=2).float()

    return x, y


def make_rings(n_per_class: int = 500, noise: float = 0.01, 
               n_rings: int = 2, base_radius: float = 2.0,
               device: torch.device | str = "cpu"):
    """
    Generate 2D points of 2 different class arranged in concentric rings.

    Args
    ----
    n_per_class: number of points per class (total = 2 * n_per_class).
    noise: standard deviation of Gaussian noise added to each point.
    n_rings: how many rings (if n_rings is even, the inner circle belongs to the opposite class of the outer ring).
    device: torch device for the returned tensors.

    Returns
    -------
    x: Tensor of shape (2*n_per_class, 2) - the coordinates.
    y: Tensor of shape (2*n_per_class, 2) - one-hot class labels (float).

    """
    theta = torch.linspace(0, 2 * torch.pi, n_per_class, device=device)
    
    def get_radii(start_ring: int): # We want class 0 to take rings 1, 3, 5... and class 1 to take 2, 4, 6...
        ring_indices = torch.arange(start_ring, n_rings + 1, 2, device=device)
        repeats = (n_per_class // len(ring_indices)) + 1
        return (ring_indices.repeat(repeats)[:n_per_class]).float()

    base_r = base_radius  # multiplier
    
    rs0 = base_r * get_radii(start_ring=1) # Class 0 radii (1, 3, 5...) * base_r
    rs1 = base_r * get_radii(start_ring=2) # Class 1 radii (2, 4, 6...) * base_r

    x0 = torch.stack([rs0 * torch.cos(theta), rs0 * torch.sin(theta)], dim=1)
    x1 = torch.stack([rs1 * torch.cos(theta), rs1 * torch.sin(theta)], dim=1)

    x = torch.cat([x0, x1], dim=0)

    if noise > 0:
        x += torch.randn_like(x) * noise

    labels = torch.cat([
        torch.zeros(n_per_class, dtype=torch.long, device=device),
        torch.ones(n_per_class, dtype=torch.long, device=device)
    ], dim=0)

    y = F.one_hot(labels, num_classes=2).float()

    return x, y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    xy_spiral, labels_spiral = make_spiral(200, noise=0.5, turns=2)
    xy_rings, labels_rings = make_rings(200, noise=0.2, n_rings=5)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].scatter(xy_spiral[:,0], xy_spiral[:,1], 
                   c=['red' if l == 1.0 else 'lightsteelblue' for l in labels_spiral[:, 0]])
    axs[1].scatter(xy_rings[:,0], xy_rings[:,1], 
                   c=['red' if l == 1.0 else 'lightsteelblue' for l in labels_spiral[:, 0]])
    plt.show()