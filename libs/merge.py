import torch


class ResidualFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, function_input, attention, gamma):
        ctx.save_for_backward(function_input, attention, gamma)
        # x: function_input, y: attention, z: gamma
        # (z*y+1)*x
        gamma_attention = attention.mul(gamma)
        gamma_attention.add_(1)
        gamma_attention.mul_(function_input)
        return gamma_attention

    @staticmethod
    def backward(ctx, grad_output):
        x, y, z = ctx.saved_tensors  # skipcq
        dz = x.mul(grad_output)  # skipcq
        dy = dz.mul(z)  # skipcq
        dx = y.mul(z)  # skipcq
        dx.add_(1)
        dx.mul_(grad_output)
        dz.mul_(x)
        return dx, dy, dz


# skipcq
residual_function = ResidualFunction.apply


class ResModule(torch.nn.Module):
    def __init__(self, residual_module, layer_module, m=0):
        super(ResModule, self).__init__()
        self.residual_module = residual_module
        self.layer_module = layer_module
        self.gamma = torch.nn.Parameter(torch.ones((1, 1)))
        torch.nn.init.orthogonal_(self.gamma.data)
        self.gamma.data.add_(m + 1)

    def forward(self, function_input, layer_input=None, scale=None):
        args = [function_input] if layer_input is None else [layer_input]
        if scale is not None:
            args.append(scale)
        res = self.residual_module(function_input)
        layer_out = self.layer_module(*args)
        gamma = self.gamma.view(*[1] * len(layer_out.size())).expand_as(layer_out)
        return residual_function(res, layer_out, gamma)
