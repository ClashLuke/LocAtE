import torch

from .conv import RevConvFunction


class SelfAttentionFunction(RevConvFunction):
    @staticmethod
    def calc(block_input, *args) -> torch.FloatTensor:
        weight0, bias0, weight1, bias1 = args
        conv = torch.nn.functional.conv1d
        batch, features, *size = block_input.size()
        block_input = block_input.view(batch, features, -1)

        block_input = RevConvFunction.conv(block_input, conv, 0, 1,
                                           weight0, bias0)
        block_input = RevConvFunction.conv(block_input, conv, 0, 1,
                                           weight1, bias1)

        block_input = block_input.softmax(-1)
        block_input = block_input.view(batch, features, *size)

        return block_input

    @staticmethod
    def forward(ctx, block_input, *args):  # skipcq
        ctx.save_for_backward(block_input, *args)
        with torch.no_grad():
            block_input = SelfAttentionFunction.calc(block_input, *args)
        block_input.requires_grad_(True)
        return block_input

    @staticmethod
    def backward(ctx, grad_output):
        args = ctx.saved_tensors
        with torch.enable_grad():
            for a in args:
                a.requires_grad_(True)
            block_output = SelfAttentionFunction.calc(*args)
            grad = torch.autograd.grad(block_output, args, grad_output)
        return grad


self_attention_function = SelfAttentionFunction.apply  # skipcq


class SelfAttention(torch.nn.Module):
    def __init__(self, features):
        super(SelfAttention, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, features, features, 1))
        self.zero_bias = torch.zeros(features)

    def forward(self, function_input):
        weight0, weight1 = self.weight
        return self_attention_function(function_input, weight0, self.zero_bias,
                                       weight1, self.zero_bias)
