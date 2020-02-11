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
        x_2 = x.mul(2)
        x_square_1 = x.pow(2)
        x_square_1.add_(1)
        lower_cosh = x_2.cosh()
        lower_cosh.add_(1)
        lower = x_square_1.pow(2/ROOTTANH_GROWTH)
        lower.mul_(lower_cosh)

        x_square_1.mul_(x.cosh().pow(2))
        x_square_1.mul_(4)
        x_square_1.div_(lower_cosh)
        x_2.mul_(x_2.sinh())
        x_2.div_(ROOTTANH_GROWTH)
        x_square_1.add_(x_2)
        x_square_1.div_(lower)

        x_square_1.mul_(grad_output)

        return x_square_1.float()
        

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
