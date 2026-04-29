from collections import OrderedDict
from typing import Callable, Type

import torch
import torch.nn as nn

from invertibles.nonlinearities import *
from invertibles.linears import LinearBlock
from invertibles.abstract_invertibles import InvertibleLayer


class InvertibleMLP(nn.Module):
    activations: dict[str, type[nn.Module]] = {
        'sigmoid': I_Sigmoid,
        'tanh': I_Tanh,
        'cubic': I_Cubic,
        'cuberoot': I_CubicRoot,
        'bilog': I_BiLog,
        'slideq': I_SlideQ,
        'arctan': I_Arctan,
        'leaky_relu': I_LeakyReLU,
        'softplus': I_SoftPlus
    }

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 2,
        hidden_width: int = 3,
        hidden_depth: int = 4,
        non_linearity: str | type[nn.Module] = 'sigmoid',
        dtype: torch.dtype = torch.float64,
        **kwargs
    ) -> None:
        """
        Invertible multilayer perceptron

        Parameters
        ----------
        input_dim: 
            number of features in input.
        output_dim: 
            dimension of the output (i.e., number of classes)
        hidden_width: 
            number of units in a single hidden layer.
        hidden_depth: 
            number of hidden layers.
        non_linearity: 
            which activation function(s) to use in each hidden layer, 
            constant throughout the network, either a string or a custom module
        """
        if hidden_width < input_dim:
            raise ValueError("Hidden dimension is not meant to be smaller than input dimension.")
        super().__init__()
        self.dtype = dtype
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_width = hidden_width
        self.hidden_depth = hidden_depth

        self.net = self.__make_net(non_linearity, **kwargs)

    def __make_net(self, non_linearity: str | type[nn.Module], **kwargs) -> nn.Sequential:
        """Internal use function that creates the net"""
        
        if isinstance(non_linearity, str): 
            if non_linearity in self.activations:
                activation = self.activations[non_linearity]
            else:
                raise ValueError(f"Invalid named non-linearity. It must be one of the following:" 
                                f" {list(self.activations.keys())}")  
        elif isinstance(non_linearity, type) and issubclass(non_linearity, nn.Module):
            if (hasattr(non_linearity, 'inverse') and hasattr(non_linearity, 'forward')):
                activation = non_linearity
            else:
                raise NotImplementedError("non_linearity must have both forward() and inverse() implemented.")
        else:
            raise ValueError("Invalid argument for non_linearity.")
        
    
        layers: list[tuple[str, nn.Module]] = [
            ('l1', LinearBlock(self.input_dim, self.hidden_width, dtype=self.dtype)),
            ('h1', activation(**kwargs))
        ]
        for depth in range(2, self.hidden_depth+1):
            layers.append((f"l{depth}", LinearBlock(self.hidden_width, self.hidden_width, dtype=self.dtype)))
            layers.append((f"h{depth}", activation(**kwargs)))
        layers.append(('out', LinearBlock(self.hidden_width, self.output_dim, dtype=self.dtype)))
        # layers.append(('phat', nn.Sigmoid()))
        return nn.Sequential(OrderedDict(layers))
    

    def forward(self, x):
        """Forward pass"""
        return self.net(x)
    

    @torch.no_grad()
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """Iterative call of the inverse method of each single layer.
        The computation of the inversion begins from the second-to-last layer.

        Parameters
        ----------
        y: a tensor that has last dimension equal to the hidden width.

        Returns
        -------
        x: the original x that originated y, if y belongs to the images of the model
        """
        layer = self.net[-2]
        assert isinstance(layer, InvertibleLayer), f"Layer {layer} is not invertible"
        x = layer.inverse(y)

        for depth in range(3, 2*self.hidden_depth+2):
            layer = self.net[-depth]
            assert isinstance(layer, InvertibleLayer), f"Layer {layer} is not invertible"
            x = layer.inverse(x)

        return x
