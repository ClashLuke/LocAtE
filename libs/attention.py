import torch

from .activation import NonLinear
from .spectral_norm import SpectralNorm


class SelfAttention(torch.nn.Module):
    def __init__(self, features, dim):
        super(SelfAttention, self).__init__()

        def kwargs(mul0, mul1):
            return dict(kernel_size=3, padding=1,
                        in_channels=features * mul0, out_channels=features * mul1)

        conv = getattr(torch.nn, f'Conv{dim}d')
        self.nlin_in = NonLinear()
        self.conv_0 = SpectralNorm(conv(**kwargs(1, 2)))
        self.nlin_0 = NonLinear()
        self.conv_1 = SpectralNorm(conv(**kwargs(1, 1)))
        self.nlin_1 = torch.nn.Softmax(dim=-1)
        self.conv_2 = SpectralNorm(conv(**kwargs(2, 2), groups=2))

    def forward(self, function_input):
        batch, features, *size = function_input.size()
        out = self.nlin_in(function_input)
        c_out = self.conv_0(out)
        c_out = self.nlin_0(c_out)
        c_out0, c_out1 = c_out.chunk(2, 1)
        m_out = self.conv_1(c_out0 + c_out1)
        m_out = m_out.view(batch, features, -1)
        m_out = self.nlin_1(m_out)
        m_out = m_out.view(batch, features, *size)
        c_out = torch.cat([m_out * c_out0, m_out * c_out1], dim=1)
        out = self.conv_2(c_out)
        out0, out1 = out.chunk(2, 1)
        out = out0 + out1
        return out
