"""This module contains some invertible non linearities, such as:
- x**3
- LeakyReLU, max(ax, x), 0<a<1
- Sigmoid, exp(x)/(1+exp(x))
- Softplus, 
"""
import torch
import torch.nn.functional as F


class I_Cubic(torch.nn.Module):
    """x**3 activation"""
    def __init__(self, slope=1e-1) -> None:
        super().__init__()
        self.slope = slope

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.pow(self.slope * inp, 3.0)

    @torch.no_grad()
    def inverse(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.pow(inp / self.slope, 1/3.0)


class I_SoftPlus(torch.nn.Module):
    """log(1+exp(x)) activation"""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.log(1 + torch.exp(inp))

    @torch.no_grad()
    def inverse(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.exp(inp) - 1)
    
    
class I_Sigmoid(torch.nn.Module):
    """exp(x)/(1_exp(x)) activation"""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(inp)
    
    @torch.no_grad()
    def inverse(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.logit(inp)
    

class I_Tanh(torch.nn.Module):
    """(exp(x)-exp(-x))/(exp(x)+exp(-x)) aka tanh(x) activation"""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.tanh(inp)
    
    @torch.no_grad()
    def inverse(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.atanh(inp)
    

class I_Arctan(torch.nn.Module):
    """arctan(x) activation"""   
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.arctan(inp)
    
    @torch.no_grad()
    def inverse(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.tan(inp) 


class I_LeakyReLU(torch.nn.Module):
    """max(ax, x), 0<a<1 activation"""
    def __init__(self, negative_slope: float = 1e-2,
                 inplace: bool = True) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

        if torch.isclose(torch.tensor(self.negative_slope), torch.tensor(0.0)):
            raise ValueError(
                "Negative slope is close to 0: {negative_slope}\n" +
                "Inverse might be unstable.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.leaky_relu(
            input,
            self.negative_slope,
            self.inplace
        )

    @torch.no_grad()
    def inverse(self, input: torch.Tensor) -> torch.Tensor:
        clone_input = input
        if not self.inplace:
            clone_input = input.clone()
        mask = clone_input < 0
        clone_input[mask] = clone_input[mask] / self.negative_slope
        return clone_input
