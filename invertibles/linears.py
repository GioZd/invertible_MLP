import torch
import torch.nn as nn

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.block = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor):
        return self.block(x) 
    
    @torch.no_grad()
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        # y shape: [batch, out_features]
        # W shape: [out_features, in_features]

        W = self.block.weight
        b = self.block.bias
        
        y_tilde = y - b
        
        diff = self.block.out_features - self.block.in_features

        if diff == 0:
            solution = torch.linalg.solve(W, y_tilde.T)
            return solution.T
        else:
            solution = torch.linalg.lstsq(W, y_tilde.T).solution
            return (solution.T)  # Shape: [batch, in_features]
