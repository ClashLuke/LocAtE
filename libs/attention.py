import torch

from .conv import ActivatedBaseConv


def feature_attention(in_size, features, dim=2, input_tensor_list=None,
                      append_output=True):
    if input_tensor_list is None:
        input_tensor_list = []
    layers = [ActivatedBaseConv(input_tensor_list, features, features, False,
                                dim, in_size, 1, 0, separable=True,
                                append_output=append_output),
              torch.nn.Softmax(dim=1)]
    return torch.nn.Sequential(*layers)


class SelfAttention(torch.nn.Module):
    def __init__(self, features, dim, input_tensor_list, append_output=True):
        super(SelfAttention, self).__init__()
        self.base_conv = ActivatedBaseConv(input_tensor_list, features, features, False,
                                           dim, 1, 1, 0, append_output=append_output)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, function_input):
        batch, features, *size = function_input.size()
        output = self.base_conv(function_input)
        output = output.view(batch, features, -1)
        output = self.softmax(output)
        output = output.view(batch, features, *size)
        return output
