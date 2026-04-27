import getopt
import os
import sys
from typing import Any, Callable

from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
# from tqdm import tqdm

from invertible_nn import InvertibleMLP
from datasets import make_spiral, make_rings
from timer import timer

PATH = './models/'
DATASETS = ['spiral', 'rings']

@timer
def train(
    max_epochs: int,
    model: InvertibleMLP,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any, # torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau,
    criterion: nn.Module | Callable = nn.CrossEntropyLoss(),
    device: torch.device = torch.device('cpu'),
) -> tuple[InvertibleMLP, list[float], list[float]]:
    model.train()
    model.to(device)
    avg_losses = []
    lrates = []
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
        avg_losses.append(avg_loss)
        lrates.append(current_lr)
        print(f'Epoch {epoch: 5d}, Average loss {avg_loss:.6f}, η {current_lr:.3e}', 
              end= '\n' if epoch % (20 if max_epochs<=100 else (100 if max_epochs<=500 else 500)) == 0 else '\r')

    return model, avg_losses, lrates

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

@torch.no_grad()
def invertibility_check(model: InvertibleMLP, 
                        test_loader: DataLoader, 
                        device: torch.device = torch.device('cpu')) -> None:
    model.eval()
    clamped_model = model.net[:-1]
    clamped_model.eval()

    mean_error = 0.0
    max_error = torch.tensor(0.0)
    total = 0
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device).float()
        output = clamped_model(x)

        # Uncomment the following lines for sanity checks on inversion steps

        # if i==0:
        #     x_tmp = output
        #     print("Final manifold:\n", x_tmp)
        #     for d, submod in enumerate(reversed(clamped_model), 1):
        #         print(f"depth = -{d}:")
        #         x_tmp = submod.inverse(x_tmp)
        #         print(x_tmp)
        
        x_tilde = model.inverse(output)
        mean_error += torch.sum((x-x_tilde)**2)
        max_error = max(max_error, torch.amax(torch.sum((x-x_tilde)**2, 1)))
        total += y.size(0)
    mean_error = mean_error/total
    print(f'Mean squared error on reconstruction: {mean_error:3.4f}')
    print(f'Max squared error on reconstruction: {max_error:3.4f}')
        


def train_spiral(epochs: int = 100, width: int = 3, depth: int = 4, 
                 lr: float = 0.015625, batch_size: int = 128, 
                 activation='softplus', folder: str | None = None, 
                 seed: int = 42, **kwargs):
    """Train, validation and visualization on the spiral toy dataset.
    Set folder=PATH to save the trained model. Name is assigned automatically on the base
    of the hyperparameters.
    """
    act = activation if isinstance(activation, str) else activation.__name__
    if folder:
        model_path = os.path.join(folder, f"spiral-w{width}d{depth}-{act}.pth")
        losses_path = os.path.join(folder, f"spiral-w{width}d{depth}-{act}-loss.dat")

    xy, labels = make_spiral(1024, noise=0.2, turns=1.5, seed=seed)
    spiral = TensorDataset(xy, labels)
    torch.manual_seed(seed)
    spiral_train, spiral_test = random_split(spiral, [0.8, 0.2], 
                                             torch.Generator().manual_seed(42))
    train_loader = DataLoader(spiral_train, batch_size=batch_size, 
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(spiral_test, batch_size=5, shuffle=False)
    mlp = InvertibleMLP(2, 2, hidden_width=width, hidden_depth=depth, 
                        non_linearity=activation, **kwargs)
    # mlp = InvertibleMLP(2, 2, hidden_width=width, hidden_depth=depth, non_linearity='tanh')
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           factor=0.75, 
                                                           patience=10, 
                                                           threshold=1e-4,
                                                           threshold_mode='abs',
                                                           cooldown=20,
                                                           min_lr=5e-6)
    mlp, losses, learning_rates = train(
        max_epochs=epochs, 
        model=mlp, 
        train_loader=train_loader, 
        optimizer=optimizer, 
        scheduler=scheduler,
        criterion=nn.CrossEntropyLoss()
    )
    # mlp.inverse(torch.tensor([[0,0.5,0,0.5,0.1],[0.5,0.1,0.5,0,0.5]]))
    test(mlp, test_loader)
    invertibility_check(mlp, test_loader)

    if folder:
        torch.save(mlp.state_dict(), model_path)
        torch.tensor(losses).numpy().tofile(losses_path, sep='\n')

    return mlp


def train_rings(epochs: int = 100, width: int = 3, depth: int = 4, 
                 lr: float = 0.015625, batch_size: int = 128, 
                 activation='softplus', folder: str | None = None, 
                 seed: int = 42, **kwargs):
    """Train, validation and visualization on the rings toy dataset.
    Set folder=PATH to save the trained model. Name is assigned automatically on the base
    of the hyperparameters.
    """
    act = activation if isinstance(activation, str) else activation.__name__
    if folder:
        model_path = os.path.join(folder, f"rings-w{width}d{depth}-{act}.pth")
        losses_path = os.path.join(folder, f"rings-w{width}d{depth}-{act}-loss.dat")

    xy, labels = make_rings(1024, base_radius=3.0, noise=0.4, n_rings=3, seed=seed)
    rings = TensorDataset(xy, labels)
    torch.manual_seed(seed)
    rings_train, rings_test = random_split(rings, [0.8, 0.2], 
                                             torch.Generator().manual_seed(42))
    train_loader = DataLoader(rings_train, batch_size=batch_size, 
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(rings_test, batch_size=5, shuffle=False)
    mlp = InvertibleMLP(2, 2, hidden_width=width, hidden_depth=depth, 
                        non_linearity=activation, **kwargs)

    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           factor=0.75, 
                                                           patience=10, 
                                                           threshold=1e-4,
                                                           threshold_mode='abs',
                                                           cooldown=20,
                                                           min_lr=5e-6)
    mlp, losses, learning_rates = train(
        max_epochs=epochs, 
        model=mlp, 
        train_loader=train_loader, 
        optimizer=optimizer, 
        scheduler=scheduler,
        criterion=nn.CrossEntropyLoss()
    )
    # mlp.inverse(torch.tensor([[0,0.5,0,0.5,0.1],[0.5,0.1,0.5,0,0.5]]))
    test(mlp, test_loader)
    invertibility_check(mlp, test_loader)

    if folder:
        torch.save(mlp.state_dict(), model_path)
        torch.tensor(losses).numpy().tofile(losses_path, sep='\n')

    return mlp


@torch.no_grad()
def draw_model(data: torch.Tensor, labels: torch.Tensor, model: nn.Module, 
               device: torch.device = torch.device('cpu')):
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
    probs = F.softmax(predictions, dim=1)
    # 3. Reshape for plotting (Transpose to align with meshgrid/cartesian_prod)
    Z = probs[:,1].view(res, res).T.numpy()
    X, Y = x_range.numpy(), y_range.numpy()
    
    plt.figure(figsize=(10, 8))
    
    # Draw the heatmap
    mesh = plt.pcolormesh(X, Y, Z, cmap='RdBu_r', alpha=0.6, shading='auto')
    plt.colorbar(mesh, label='Probability for class 1')
    
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
    args = sys.argv[1:]
    options = "D:w:d:a:s:"
    long_options = ["dataset=", "width=", "depth=", 
                    "activation=", "kwargs=", "seed=",
                    "no-plot"]
    # defaults
    dataset = 'spiral'
    width = 5
    depth = 4
    activation = 'bilog'
    kwargs = {}
    seed = 7
    plot = True
    try:
        arguments, values = getopt.getopt(args, options, long_options)
        for currentArg, currentVal in arguments:
            if currentArg in ("-D", "--dataset"):
                dataset = currentVal.lower()
            elif currentArg in ("-w", "--width"):
                width = int(currentVal)
            elif currentArg in ("-d", "--depth"):
                depth = int(currentVal)
            elif currentArg in ("-a", "--activation"):
                activation = currentVal.lower()
            elif currentArg in ("--kwargs",):
                for kwval in currentVal.split(','):
                    kw, val = kwval.split('=')
                    kwargs[kw] = float(val)
            elif currentArg in ("-s", "--seed"):
                seed = int(currentVal)
            elif currentArg in ("--no-plot",):
                plot = False
    except getopt.error as err:
        raise getopt.GetoptError(str(err))
    

    if dataset == 'spiral':
        print("Training `spiral`...")
        xy, labels = make_spiral(1024, noise=0.2, turns=1.5, seed=seed)
        model = train_spiral(epochs=1001, width=width, depth=depth, lr=0.005, 
                             batch_size=128, activation=activation, folder=PATH, 
                             seed=seed, **kwargs)  
        print("...model and loss history saved!")    
    elif dataset == 'rings':
        print("Training `rings`...")
        xy, labels = make_rings(1024, base_radius=3.0, noise=0.4, 
                                n_rings=3, seed=seed)
        model = train_rings(epochs=1001, width=width, depth=depth, 
                            lr=0.005, batch_size=128, activation=activation, 
                            folder=PATH, seed=seed, **kwargs)
        print("...model and loss history saved")
    else:
        print(f"Dataset `{dataset}` is currently not available. Try one of the following:", 
              *DATASETS)
        exit()

    if plot:
        draw_model(xy, labels, model)
