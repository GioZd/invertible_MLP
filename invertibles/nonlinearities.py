"""This module contains some invertible non linearities, such as:
- x**3
- LeakyReLU, max(ax, x), 0<a<1
- Sigmoid, exp(x)/(1+exp(x))
- Softplus, 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class I_Cubic(nn.Module):
    """x**3 activation."""
    def __init__(self, slope=1e-1) -> None:
        super().__init__()
        self.slope = slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.slope * x**3.0

    @torch.no_grad()
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.sign(y)* ( (torch.abs(((1/self.slope) * y)) ** (1./3.)))
    

class I_CubicRoot(nn.Module):
    """x**(1/3) activation."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * (torch.abs(x) ** (1./3.))

    @torch.no_grad()
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y ** 3.0
    

class I_BiLog(nn.Module):
    """log(1+x) for positive entries, -log(1-x) for negative entries."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x)*torch.log(torch.abs(x)+1.0)

    @torch.no_grad()
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.sign(y)*(torch.exp(torch.abs(y))-1)
    

class I_SlideQ(nn.Module):
    """Decays like logarithm for negative inputs, 
    increases quadratically for positive inputs.
    """
    def __init__(self) -> None:
        super().__init__()
        self.k = 0.5
        self.a = 0.125
        self.c = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            (x>=0)*(self.c*x**2+self.a*self.k*x) # positive part
            + (x<0)*(-self.k*torch.log(self.a*torch.abs(x)+1)) # negative part
        )

    @torch.no_grad()
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        ak = self.a*self.k
        return (
            (y>=0)*(-ak+torch.sqrt(ak**2+4*self.c*torch.abs(y)))/(2*self.c) +
            (y<0)*(-(torch.exp(-y/self.k)-1)/self.a)
        )
    
class I_SlideL(nn.Module):
    """-log(-x+1) on the left of 0, x on the right"""
    def __init__(self) -> None:
        super().__init__() 


class I_SoftPlus(nn.Module):
    """log(1+exp(x)) activation."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.log(1 + torch.exp(inp))

    @torch.no_grad()
    def inverse(self, inp: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.exp(inp) - 1)
    
    
class I_Sigmoid(nn.Module):
    """exp(x)/(1_exp(x)) activation."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)
    
    @torch.no_grad()
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.logit(y)
    

class I_Tanh(nn.Module):
    """(exp(x)-exp(-x))/(exp(x)+exp(-x)) aka tanh(x) activation."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)
    
    @torch.no_grad()
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.atanh(y)
    

class I_Arctan(nn.Module):
    """arctan(x) activation."""   
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.arctan(x)
    
    @torch.no_grad()
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.tan(y) 


class I_LeakyReLU(nn.Module):
    """max(ax, x), 0<a<1 activation."""
    def __init__(self, negative_slope: float = 1e-2,
                 inplace: bool = True) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

        if torch.isclose(torch.tensor(self.negative_slope), torch.tensor(0.0)):
            raise ValueError(
                "Negative slope is close to 0: {negative_slope}\n" +
                "Inverse might be unstable.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.leaky_relu(
            x,
            self.negative_slope,
            self.inplace
        )

    @torch.no_grad()
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        clone_input = y.clone() if not self.inplace else y
        # mask = clone_input < 0
        # clone_input[mask] = clone_input[mask] / self.negative_slope
        clone_input = torch.amin(torch.stack([clone_input, clone_input/self.negative_slope]), dim=0)
        return clone_input


if __name__ == '__main__':
    x = torch.tensor([[-1.0, 1.0],[-2.5, 0.0]], dtype=torch.double)
    cubic = I_Cubic(slope=0.2)
    cuberoot = I_CubicRoot()
    bilog = I_BiLog()
    slideq = I_SlideQ()
    softplus = I_SoftPlus()
    sigmoid = I_Sigmoid()
    tanh = I_Tanh()
    arctan = I_Arctan()
    lrelu = I_LeakyReLU(negative_slope=0.2)

    print("Sanity check on various functions...")

    assert torch.all(torch.isclose(x, cubic.inverse(cubic(x)))), f"Cubic activation is not working."
    assert torch.all(torch.isclose(x, cuberoot.inverse(cuberoot(x)))), f"Cube root activation is not working."
    assert torch.all(torch.isclose(x, bilog.inverse(bilog(x)))), f"Two-sided logarithmic activation is not working."
    assert torch.all(torch.isclose(x, slideq.inverse(slideq(x)))), f"Mixed logarithmic-quadratic activation is not working."
    assert torch.all(torch.isclose(x, softplus.inverse(softplus(x)))), f"Softplus activation is not working."
    assert torch.all(torch.isclose(x, sigmoid.inverse(sigmoid(x)))), f"Sigmoid activation is not working."
    assert torch.all(torch.isclose(x, tanh.inverse(tanh(x)))), f"Tanh activation is not working."
    assert torch.all(torch.isclose(x, arctan.inverse(arctan(x)))), f"Arctan activation is not working."
    assert torch.all(torch.isclose(x, lrelu.inverse(lrelu(x)))), f"Leaky ReLU activation is not working."

    print("All functions seem correctly implemented!")