from typing import Callable

from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from invertible_nn import InvertibleMLP
from datasets import make_spiral, make_rings

def train(
    max_epochs: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau,
    criterion: nn.Module | Callable = nn.CrossEntropyLoss(),
    device: torch.device = torch.device('cpu'),
):
    model.train()
    model.to(device)

    for epoch in range(max_epochs):
        total_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device).float()
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
          
            total_loss += loss.item()
            print(f'Epoch {epoch: 5d}, Batch {i: 3d}, Loss {loss.item():.6f}', end='\r')

        avg_loss = total_loss / len(train_loader)
        scheduler.step(metrics=avg_loss) 
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch: 5d}, Average loss {avg_loss:.6f}, η {current_lr:.3e}', 
              end= '\n' if epoch % (20 if max_epochs<=100 else (100 if max_epochs<=500 else 500)) == 0 else '\r')

    return model


@torch.no_grad()
def test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device = torch.device('cpu')
) -> float:

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x, y in test_loader:
        x, y = x.to(device), y.to(device).float()
        output = model(x)
        # predicted_labels = (output>0.5).float().cpu().numpy()
        predicted_labels = torch.argmax(output, dim=1)
        true_labels = torch.argmax(y, dim=1)

        total += y.size(0)
        correct += (true_labels == predicted_labels).sum().item()

        all_preds.extend(predicted_labels)
        all_labels.extend(true_labels)

    accuracy = correct / total * 100
    print(f'Overall Accuracy: {accuracy:.2f}%')

    print("\nClassification Report:\n", 
        classification_report(
            all_labels,
            all_preds,
            digits=4
        )
    )

    return accuracy

def train_spiral(epochs: int = 100, width: int = 3, depth: int = 4, 
                 lr: float = 0.015625, batch_size: int = 128):
    xy, labels = make_spiral(1024, noise=0.2, turns=1.5)
    # xy, labels = make_rings(1024, noise=0.1, n_rings=4, base_radius=2.5)
    # spiral = TensorDataset(xy, labels[:,0])
    spiral = TensorDataset(xy, labels)
    spiral_train, spiral_test = random_split(spiral, [0.8, 0.2], torch.Generator().manual_seed(42))
    train_loader = DataLoader(spiral_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(spiral_test, batch_size=4096, shuffle=False)
    mlp = InvertibleMLP(2, 2, hidden_width=width, hidden_depth=depth, non_linearity='leaky_relu', negative_slope=0.1)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           factor=0.75, 
                                                           patience=10, 
                                                           threshold=1e-4,
                                                           threshold_mode='abs',
                                                           cooldown=20,
                                                           min_lr=1e-5)
    train(
        max_epochs=epochs, 
        model=mlp, 
        train_loader=train_loader, 
        optimizer=optimizer, 
        scheduler=scheduler,
        criterion=nn.CrossEntropyLoss()
    )
    mlp.inverse(torch.tensor([[0,0.5,0,0.5,0.1],[0.5,0.1,0.5,0,0.5]]))
    test(mlp, test_loader)
    return mlp

@torch.no_grad()
def draw_model(data: torch.Tensor, labels: torch.Tensor, model: nn.Module, device: torch.device = torch.device('cpu')):
    import matplotlib.pyplot as plt
    model.eval()
    model.to(device)
    
    # 1. Create a dense grid
    res = 200 
    x_range = torch.linspace(-15, 15, res)
    y_range = torch.linspace(-15, 15, res)
    grid_points = torch.cartesian_prod(x_range, y_range).to(device)
    
    # 2. Get predictions
    predictions = model(grid_points).squeeze().cpu()
    
    # 3. Reshape for plotting (Transpose to align with meshgrid/cartesian_prod)
    Z = predictions[:,1].view(res, res).T.numpy()
    X, Y = x_range.numpy(), y_range.numpy()
    
    plt.figure(figsize=(10, 8))
    
    # Draw the heatmap
    mesh = plt.pcolormesh(X, Y, Z, cmap='RdBu_r', alpha=0.6, shading='auto')
    plt.colorbar(mesh, label='Logit for class 1')
    
    # --- NEW: Add the Decision Boundary Line ---
    # levels=[0.5] draws a line exactly where the probability is 0.5
    plt.contour(X, Y, Z, levels=[0.5], colors='black', linewidths=2)
    
    # Draw the scattered data points
    # Using labels[:, 0] specifically to handle the shape from make_spiral
    scatter_colors = ['red' if l == 1.0 else 'blue' for l in labels[:,1].view(-1)]
    plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), 
                c=scatter_colors, edgecolors='white', s=25, zorder=3)
    
    plt.title("Decision Boundary and Prediction Heatmap")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.show()

if __name__ == '__main__':
    torch.manual_seed(7)
    xy, labels = make_spiral(1024, noise=0.2, turns=1.5)
    # xy, labels = make_rings(1024, noise=0.1, n_rings=3, base_radius=2.5)
    model = train_spiral(epochs=401, width=5, depth=4, lr=0.005, batch_size=256)
    draw_model(xy, labels, model)