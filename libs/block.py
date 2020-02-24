import torch
from torch import nn

from .attention import SelfAttention
from .config import (ATTENTION_EVERY_NTH_LAYER, DEPTH, INPUT_VECTOR_Z,
                     MIN_ATTENTION_SIZE)
from .conv import DeepResidualConv
from .linear import LinearModule
from .merge import ResModule
from .scale import Scale
from .spectral_norm import SpectralNorm
from .utils import prod


class Block(nn.Module):
    def __init__(self, in_size, in_features, out_features, stride, transpose,
                 block_number, dim=2):
        super().__init__()
        self.input_tensor_list = []
        self.attention = (in_size >= MIN_ATTENTION_SIZE and
                          block_number % ATTENTION_EVERY_NTH_LAYER == 0)
        self.scale_layer = Scale(in_features, out_features, stride, transpose, dim=dim)
        self.res_module_i = DeepResidualConv(out_features, out_features, False, 1, True,
                                             dim, DEPTH, self.input_tensor_list, True,
                                             True)
        if self.attention:
            self.res_module_s = ResModule(lambda x: x,
                                          SpectralNorm(
                                                  SelfAttention(out_features)))

    def forward(self, function_input, scales=None):
        if scales is None:
            scales = [None] * 4
        scaled = self.scale_layer(function_input)
        out = self.res_module_i(scaled, scale=scales[0:2])
        if self.attention:
            out = self.res_module_s(out, scale=scales[2])
        return out


class BlockBlock(nn.Module):
    def __init__(self, block_count, in_size, features, strides, transpose,
                 mul_channel=False, dim=2):
        super().__init__()
        self.block_count = block_count

        factors = strides if transpose else [1 / s for s in strides]

        def feature_tuple(idx):
            return features[idx], features[idx + 1]

        def size(idx):
            out = in_size * prod(factors[:idx + 1])
            nout = int(out + 1 - 1e-12)
            return nout

        blocks = [Block(size(i), *feature_tuple(i), strides[i], transpose, i, dim=dim)
                  for i in range(block_count)]
        self.blocks = blocks

        for i, block in enumerate(blocks):
            setattr(self, f'block_{i}', block)
        sums = [0]
        depths = []
        if mul_channel:
            mul_blocks = []
            for i in range(block_count):
                scales = 2 + int(blocks[i].attention)
                depths.append(scales)
                sums.append(sums[-1] + scales)
                inp, out = feature_tuple(i)
                mul_blocks.append(LinearModule(inp + INPUT_VECTOR_Z * bool(i), out))
                attention_scales = [LinearModule(out + INPUT_VECTOR_Z, out) for _ in
                                    range(1, scales)]
                mul_blocks.extend(attention_scales)

            self.mul_blocks = mul_blocks

            for i, block in enumerate(mul_blocks):
                setattr(self, f'mul_block_{i}', block)

        self.depths = depths
        self.sums = sums
        self.out_features = feature_tuple(block_count - 1)[1]

    def forward(self, function_input, noise=None):
        next_input = None
        for i in range(self.block_count):
            if noise is None:
                operand = None
            else:
                operand = []
                for idx in range(self.depths[i]):
                    if next_input is None:
                        next_input = noise
                    else:
                        next_input = torch.cat([noise, next_input], dim=1)
                    next_input, factor = self.mul_blocks[self.sums[i] + idx](next_input)
                    operand.append(factor.view(*factor.size(), 1, 1))
            function_input = self.blocks[i](function_input, scales=operand)
        return function_input
