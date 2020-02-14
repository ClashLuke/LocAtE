import torch
from torch import nn
from .config import device, ROOTTANH_GROWTH


class RootTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input = input.double()
        ctx.save_for_backward(input)
        output = input.pow(2)
        output.add_(1)
        output.pow_(1/ROOTTANH_GROWTH)
        output.mul_(input.tanh())
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        x_2 = x.pow(2)
        x_2.add_(1)
        sech_2_x = x.cosh()
        sech_2_x.pow_(2)
        sech_2_x.reciprocal_()
        sech_2_x.mul_(x_2)
        sech_2_x.mul_(2)
        tanh_x = x.tanh()
        tanh_x.mul_(x)
        sech_2_x.add_(tanh_x)
        x_2.pow_((ROOTTANH_GROWTH-1)/ROOTTANH_GROWTH)
        x_2.mul_(2)
        sech_2_x.div_(x_2)
        sech_2_x.mul_(grad_output)
        return sech_2_x.float()
        
        
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.slope = torch.nn.Parameter(torch.ones(1))

    def forward(self, input):
        return input.mul(torch.sigmoid(torch.mul(input, self.slope.data)))


class RootTanhModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = RootTanh.apply

    def forward(self, input):
        return self.fn(input)


nlinear = RootTanhModule
