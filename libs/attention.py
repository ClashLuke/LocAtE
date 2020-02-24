import torch

from .config import DEVICE
from .conv import RevConvFunction


class SelfAttentionFunction(RevConvFunction):
    @staticmethod
    def calc(block_input, *args) -> torch.FloatTensor:
        weight0, bias0, weight1, bias1, scale = args
        if scale is not None:
            scale = scale.view(scale.size(0), scale.size(1), 1)
        conv = torch.nn.functional.conv1d
        batch, features, *size = block_input.size()
        block_input = block_input.view(batch, features, -1)

        block_input = RevConvFunction.conv(block_input, conv, 0, 1,
                                           weight0, bias0, scale)
        block_input = RevConvFunction.conv(block_input, conv, 0, 1,
                                           weight1, bias1, scale)

        block_input = block_input.softmax(-1)
        block_input = block_input.view(batch, features, *size)

        return block_input

    @staticmethod
    def forward(ctx, block_input, *args):  # skipcq
        with torch.no_grad():
            ctx.save_for_backward(block_input, *args)
            block_input = SelfAttentionFunction.calc(block_input, *args)
        block_input.requires_grad_(True)
        return block_input

    @staticmethod
    def backward(ctx, grad_output):
        args = ctx.saved_tensors
        with torch.enable_grad():
            for a in args:
                try:
                    a.requires_grad_(True)
                except AttributeError:
                    pass
            block_output = SelfAttentionFunction.calc(*args)
            delete = args[-1] is None
            if delete:
                args = args[:-1]
            grad = torch.autograd.grad(block_output, args, grad_output)
        if delete:
            grad = (*grad, None)
        return grad


self_attention_function = SelfAttentionFunction.apply  # skipcq


class SelfAttention(torch.nn.Module):
    def __init__(self, features):
        super(SelfAttention, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, features, features, 1))
        self.zero_bias = torch.zeros(features).to(DEVICE)

    def forward(self, function_input, scale=None):
        weight0, weight1 = self.weight
        return self_attention_function(function_input, weight0, self.zero_bias,
                                       weight1, self.zero_bias, scale)
