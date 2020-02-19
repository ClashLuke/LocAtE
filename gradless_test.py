import torch


class Gradless(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, *args):
        ctx.constant = len(args)
        with torch.no_grad():
            return module(*args)
    @staticmethod
    def backward(ctx, grad_output):
        return tuple([None]+[grad_output]*ctx.constant)

def get_itm(self, grad_input, grad_output):
    self.backward_out = grad_output
    return None

def backward_hook(self, grad_input, grad_output):
    self.module.register_backward_hook(get_itm)
    with torch.enable_grad():
        o = [g.clone() for g in grad_input]
        for t in o:
            t.requires_grad_(True)
        out = self.module(*o)
    out.backward(torch.ones_like(out))
    back = self.module.backward_out
    for g in grad_output:
        g.mul_(*back)
    return None


class GradlessModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.register_backward_hook(backward_hook)

    def forward(self, *args):
        return Gradless.apply(self.module, *args)


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        out = (x ** 2 + 1) ** (1 / 4) * x.tanh() * y
        return out


r0 = torch.randn((1, 10000), requires_grad=True).double()
r1 = torch.randn((1, 10000), requires_grad=True).double()
print(torch.std_mean(r1))

gradless_module = GradlessModule(Module())
gradless_module.register_backward_hook(backward_hook)
print(gradless_module)
gradcheck = torch.autograd.gradcheck
o = gradless_module(r0, r1)
print(o)
print(o.mean().backward())

