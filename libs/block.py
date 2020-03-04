import torch
from torch import nn

from .Kernelless.module import Kernelless
from .attention import SelfAttention
from .config import (ATTENTION_EVERY_NTH_LAYER, INPUT_VECTOR_Z,
                     MIN_ATTENTION_SIZE)
from .inplace_norm import Norm
from .linear import LinearModule
from .merge import ResModule
from .scale import Scale
from .utils import prod


class BigNetwork(nn.Module):
    def __init__(self, in_size, in_features, out_features, stride, transpose,
                 block_number, dim=2):
        super().__init__()
        print("T", in_features, out_features, in_size)
        conv = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
        self.conv_module = Norm(in_features, conv(in_features, out_features, 5- transpose, stride=stride, padding=2-transpose),
#                                Kernelless(in_features, out_features, in_size, stride,
#                                           dim, transpose),
                                dim)
        self.attention = (in_size >= MIN_ATTENTION_SIZE and
                          block_number % ATTENTION_EVERY_NTH_LAYER == 0)
        if self.attention:
            self.att_module = ResModule(lambda x: x, Norm(out_features,
                                                          SelfAttention(out_features,
                                                                        dim), dim))

    def forward(self, function_input, scale=None):
        if scale is None:
            scale = [None, None]
        out = self.conv_module(function_input, scale=scale[0])
        if self.attention:
            if scale is not None:
                out = self.att_module(out, scale=scale[1])
        return out


def Block(in_size, in_features, out_features, stride, transpose, block_number, dim=2):
    return ( #ResModule(Scale(in_features, out_features, stride, transpose),
                     BigNetwork(in_size, in_features, out_features, stride, transpose,
                                block_number, dim))


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
            print(out, nout)
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
            prev_out = 0
            for i in range(block_count):
                try:
                    scales = int(blocks[i].layer_module.attention)
                except:
                    scales = int(blocks[i].attention)
                depths.append(2 + scales)
                sums.append(sums[-1] + scales + 2)
                inp, out = feature_tuple(i)
                if prev_out and prev_out != inp:
                    group_inp = prev_out
                else:
                    group_inp = inp
                mul_blocks.append(
                        LinearModule(group_inp + INPUT_VECTOR_Z * bool(i), inp))
                mul_blocks.append(LinearModule(inp + INPUT_VECTOR_Z, out))

                if scales:
                    attention_scales = [LinearModule(out + INPUT_VECTOR_Z, out) for _ in
                                        range(scales)]
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
            function_input = self.blocks[i](function_input, scale=operand)
        return function_input
