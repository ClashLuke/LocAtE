import torch
from torch import nn

from .attention import SelfAttention
from .config import (ATTENTION_EVERY_NTH_LAYER, DEPTH, INPUT_VECTOR_Z,
                     MIN_ATTENTION_SIZE)
from .conv import DeepResidualConv
from .linear import LinearModule
from .merge import ResModule
from .utils import prod


class BigNetwork(nn.Module):
    def __init__(self, in_size, in_features, out_features, stride, transpose,
                 block_number, dim=2):
        super().__init__()
        self.side_path = DeepResidualConv(in_features, out_features, transpose, stride,
                                          depth=1, dim=dim)
        self.res_module_i = DeepResidualConv(in_features,
                                             out_features, transpose,
                                             stride,
                                             depth=DEPTH, dim=dim)
        if (in_size >= MIN_ATTENTION_SIZE and
                block_number % ATTENTION_EVERY_NTH_LAYER == 0):
            self.res_module_s = SelfAttention(out_features, dim)
            self.attention = True
        else:
            self.attention = False

    def forward(self, function_input, scales=None):
        if scales is not None:
            function_input = function_input * scales[0]
        out = self.res_module_i(function_input)
        if self.attention:
            if scales is not None:
                out = out * scales[1]
            out = self.res_module_s(out)
        return out


def Block(in_size, in_features, out_features, stride, transpose, block_number, dim=2):
    return ResModule(DeepResidualConv(in_features, out_features, transpose, stride,
                                      depth=1, dim=dim),
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
                scales = int(blocks[i].layer_module.attention)
                depths.append(1 + scales)
                sums.append(sums[-1] + scales + 1)
                inp, out = feature_tuple(i)
                if prev_out and prev_out != inp:
                    group_inp = prev_out
                else:
                    group_inp = inp
                mul_blocks.append(
                        LinearModule(group_inp + INPUT_VECTOR_Z * bool(i), inp))
                if scales:
                    mul_blocks.append(LinearModule(inp + INPUT_VECTOR_Z, out))
                    attention_scales = [LinearModule(out + INPUT_VECTOR_Z, out) for _ in
                                        range(1, scales)]
                    mul_blocks.extend(attention_scales)
                    prev_out = out
                else:
                    prev_out = inp

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
