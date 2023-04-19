
""" Swish. """ 

import torch
import torch.nn as nn

class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return SwishEfficient.apply(x)

class SwishEfficient(torch.autograd.Function):
    """Swish activation function: x * sigmoid(x)."""

    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))