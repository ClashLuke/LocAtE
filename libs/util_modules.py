from torch import nn

from .utils import view_expand


class Expand(nn.Module):
    def __init__(self, *target_size):
        super(Expand, self).__init__()
        self.target_size = target_size

    def forward(self, function_input):
        return view_expand(self.target_size, function_input)
