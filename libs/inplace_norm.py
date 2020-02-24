import torch


class MeanSubMulDivAdd(torch.autograd.Function):
    @staticmethod
    # skipcq
    def forward(ctx, x, y, z, w):
        ctx.save_for_backward(x, y, z)
        output = x - x.mean()
        output.mul_(y)
        output.div_(z)
        output.add_(w)
        return output

    @staticmethod
    # skipcq
    def backward(ctx, grad_output):
        x, y, z = ctx.saved_tensors
        x_g = y * grad_output / z
        x_g.sub_(x_g.mean())
        base_x = x - x.mean()
        base_x.mul_(grad_output)
        y_g = base_x / z
        z_g = y_g * y
        z_g.div_(z)
        z_g.neg_()
        return x_g, y_g, z_g, grad_output


# skipcq
mean_sub_mul_div_add = MeanSubMulDivAdd.apply


class InPlaceNorm(torch.nn.Module):
    def __init__(self, features=1, dim=2):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((1, features, *[1] * dim)))
        self.bias = torch.nn.Parameter(torch.zeros((1, features, *[1] * dim)))

    def forward(self, function_input: torch.FloatTensor, scale=None):
        if scale is None:
            return mean_sub_mul_div_add(function_input, self.weight,
                                        function_input.std(), self.bias)
        return mean_sub_mul_div_add(function_input, scale,
                                    function_input.std(), self.bias)


class Norm(torch.nn.Module):
    def __init__(self, features, module, dim=2):
        super().__init__()
        self.i_norm = InPlaceNorm(features, dim=dim)
        self.module = module

    def forward(self, function_input, scale=None):
        module_input = self.i_norm(function_input, scale)
        return self.module(module_input)
