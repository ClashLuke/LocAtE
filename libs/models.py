import torch
from torch import nn

from .block import BlockBlock
from .config import (DEVICE, DIS_FEATURES, D_STRIDE, END_LAYER, FACTOR, GEN_FEATURES,
                     G_STRIDE, IMAGE_SIZE, INPUT_VECTOR_Z, LAYERS, START_LAYER)
from .conv import DeepResidualConv
from .merge import ResModule
from .utils import get_feature_list
from .spectral_norm import SpectralNorm
from .inplace_norm import Norm

def quadnorm(number: int):
    return number // 4 * 4


class GFeatures:
    def __init__(self, base, total_layers):
        self.base = base
        self.total_layers = total_layers

    def __call__(self, idx):
        return quadnorm(int(GEN_FEATURES * FACTOR ** ((idx) - self.total_layers)))


class DFeatures:
    def __init__(self, base, total_layers):
        self.base = base
        self.total_layers = total_layers

    def __call__(self, idx):
        return quadnorm(int(DIS_FEATURES * FACTOR ** ((idx + 1) - self.total_layers)))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        clayers = LAYERS - 1
        stride_count_factor = G_STRIDE // 2
        strides = [G_STRIDE] * (clayers // stride_count_factor) + [2] * (
                clayers % stride_count_factor)

        in_f = INPUT_VECTOR_Z
        g_features = GFeatures(0, clayers)
        out_f = g_features(clayers)
        feature_list = get_feature_list(strides, True, g_features)
        if START_LAYER >= 1:
            self.input_block = DeepResidualConv(in_f, out_f, False, 1, False, 2,
                                                START_LAYER, normalize=False)
        else:
            self.input_block = lambda x: x
            out_f = in_f
        feature_list.insert(0, out_f)
        feature_list[-1] = 3
        self.conv_block = BlockBlock(len(strides), 1, feature_list, strides, True, True)
#        self.out_conv = Norm(self.conv_block.out_features, SpectralNorm(nn.Conv2d(self.conv_block.out_features, 3, 5, padding=2)), 2)

        self.g_in = feature_list[0]

        self.noise = torch.randn(1, INPUT_VECTOR_Z, 2, 2, device=DEVICE)

    def forward(self, function_input):
        expanded_noise = self.noise.expand(function_input.size(0), -1, -1, -1)
        conv_out = self.input_block(expanded_noise)
        conv_out = self.conv_block(conv_out, function_input)
 #       conv_out = self.out_conv(conv_out)
        return conv_out.tanh()


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        clayers = LAYERS - 1
        stride_count_factor = D_STRIDE // 2
        strides = [D_STRIDE] * (clayers // stride_count_factor) + [2] * (
                clayers % stride_count_factor)
        d_features = DFeatures(0, len(strides))
        feature_list = get_feature_list(strides, False, d_features)
        if END_LAYER:
            feature_list.append(feature_list[-1])
        else:
            feature_list.append(1)
        cat_module = SpectralNorm(nn.Conv2d(3, d_features(0), 5, stride=2, padding=2))
        block_block = BlockBlock(len(strides), IMAGE_SIZE, feature_list, strides,
                                 False)
        conv = [ResModule(lambda x: x,
                          DeepResidualConv(block_block.out_features,
                                           block_block.out_features, 1, False, 1, 0,
                                           1))
                for i in range(END_LAYER - 1)]
        if END_LAYER:
            conv.append(
                DeepResidualConv(block_block.out_features, 1, False, 1, False, 2, 1))
        conv.insert(0, cat_module)
        conv.insert(1, block_block)
        self.main = nn.Sequential(*conv)

    def forward(self, function_input):
        return self.main(function_input)
