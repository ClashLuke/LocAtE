from torch import nn

from .utils import view_expand


class Expand(nn.Module):
    def __init__(self, *target_size):
        super(Expand, self).__init__()
        self.target_size = target_size

    def forward(self, function_input):
        return view_expand(self.target_size, function_input)


class ViewBatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch = 1

    def forward(self, function_input):
        _, _, *size = function_input.size()
        return function_input.view(self.batch, -1, *size)


class ViewFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = 1

    def forward(self, function_input):
        _, _, *size = function_input.size()
        return function_input.view(-1, self.features, *size)
