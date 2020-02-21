from torch import nn

from .merge import CatModule
from .spectral_norm import SpectralNorm


class FeaturePooling(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features

    def forward(self, function_input):
        input_size = list(function_input.size())
        input_size[1] = self.out_features
        input_view = function_input.view(*input_size, -1)
        return input_view.mean(dim=-1)


def Scale(in_features, out_features, stride, transpose, dim=2):
    reslayers = []
    if in_features > out_features:
        reslayers.append(FeaturePooling(out_features))
    elif out_features > in_features:
        reslayers.append(CatModule(lambda x: x, SpectralNorm(
                getattr(nn, f'Conv{dim}d')(in_features, out_features - in_features,
                                           1))))
    if stride > 1:
        if transpose:
            reslayers.append(nn.Upsample(mode='bilinear', scale_factor=stride,
                                         align_corners=False))
        else:
            reslayers.append(getattr(nn, f'AvgPool{dim}d')(stride, stride))
    if len(reslayers) > 1:
        return nn.Sequential(*reslayers)
    if not reslayers:
        return lambda x: x
    return reslayers[0]
