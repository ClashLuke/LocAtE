import numpy as np

from .modules import *


def MultiInception(in_size, strides, features, invert=False, in_features=None, out_features=None, transpose=False,
                   **kwargs):
    r = range(len(strides) - 1, -1, -1) if invert else range(len(strides))
    feature_list = [features(i) for i in r]
    if in_features is not None:
        feature_list[0] = (in_features, feature_list[0][1])
    if out_features is not None:
        feature_list[-1][-1] = (feature_list[-1][0], out_features)

    def prod(arr, idx):
        return int(np.prod(arr[:idx]))

    out = [Block(in_size * prod(strides, i + 1) if transpose else in_size // prod(strides, i + 1),
                 *feature_list[i],
                 stride=s,
                 transpose=transpose) for i, s in enumerate(strides)]
    return out


def BaseMLP(in_features, out_features, activation=nn.Tanh(), dropout=False, norm=True, **kwargs):
    layers = [nn.Linear(in_features, out_features, bias=False)]
    layers.append(activation())
    if dropout:
        layers.append(nn.Dropout)
    if norm:
        layers.append(nn.BatchNorm1d(out_features))
    return layers


def MultiMLP(layers, features, invert=False, double=False, factor=lambda x: 1, sequential=False, creator=BaseMLP,
             end=-1, **kwargs):
    r = range(layers - 1, end, -1) if invert else range(layers)
    out = [creator(in_features=features(i)[0],
                   out_features=features(i)[0] if double else features(i - 1)[0],
                   in_size=factor(j),
                   **kwargs) for j, i in enumerate(r)]

    if sequential:
        try:
            out = [nn.Sequential(*MLP) for MLP in out]
        except:
            pass
    else:
        try:
            out = [L for MLP in out for L in MLP]
        except:
            pass
    return out
