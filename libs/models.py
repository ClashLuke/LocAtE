from .modules import *
from .util_modules import *
from .utils import *

quadnorm = lambda x: x // 4 * 4
D_LAYERS = 0
G_LAYERS = 0


class GFeatures:
    def __init__(self, base, layers):
        self.base = base
        self.layers = layers

    def __call__(self, idx):
        return quadnorm(int(ngf * factor ** ((idx) - self.layers)))


class DFeatures:
    def __init__(self, base, layers):
        self.base = base
        self.layers = layers

    def __call__(self, idx):
        return quadnorm(int(ndf * factor ** ((idx + 1) - self.layers)))


g_features = None
d_features = None

nz = quadnorm(input_vector_z)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        global g_in
        clayers = layers - 2
        stride_count_factor = g_stride // 2
        strides = [g_stride] * (clayers // stride_count_factor) + [2] * (clayers % stride_count_factor)
        global G_LAYERS
        global g_features
        G_LAYERS = len(strides)

        in_f = input_vector_z
        g_features = GFeatures(0, clayers)
        out_f = g_features(clayers)
        feature_list = get_feature_list(strides, True, g_features)

        self.input_block = nn.Sequential(*[ResModule(lambda x: x,
                                                     FactorizedConvModule(in_f, out_f, 3, False, 1, 1))
                                           for _ in range(start_layer)])
        if start_layer < 1:
            out_f = in_f
        feature_list.insert(0, out_f)
        self.conv_block = BlockBlock(len(strides), 2, feature_list, strides, True, True)
        self.out_conv = FactorizedConvModule(self.conv_block.out_features, 3, 5, True, 1, 2, 2, False)

        self.g_in = feature_list[0]

        self.input = torch.randn(1, input_vector_z, 2, 2, device=device)

    def forward(self, input):
        expanded_input = self.input.expand(input.size(0), -1, -1, -1)
        conv_out = self.input_block(expanded_input)
        conv_out = self.conv_block(conv_out, input)
        conv_out = self.out_conv(conv_out)
        return conv_out.tanh()


class Print(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        print(input.size())
        return input


class Discriminator(nn.Module):
    def __init__(self):
        global D_LAYERS
        global d_features
        super(Discriminator, self).__init__()
        clayers = layers - 1
        stride_count_factor = d_stride // 2
        strides = [d_stride] * (clayers // stride_count_factor) + [2] * (clayers % stride_count_factor)
        D_LAYERS = len(strides)
        d_features = DFeatures(0, D_LAYERS)
        f = d_features(len(strides))
        feature_list = get_feature_list(strides, False, d_features)
        feature_list.append(feature_list[-1])
        cat_module = ResModule(Scale(3, d_features(0), 2, False),
                               FactorizedConvModule(3, d_features(0), 5, False, 1, 2, 2, False))
        block_block = BlockBlock(len(strides), image_size // 2, feature_list, strides, False)
        conv = [ResModule(lambda x: x,
                          FactorizedConvModule(block_block.out_features, block_block.out_features, 1, False, 1, 0, 1))
                for i in range(end_layer - 1)]


        conv.append(FactorizedConvModule(block_block.out_features, 1, 1, False, 1, 0, 1, False))
        conv.insert(0, cat_module)
        conv.insert(1, block_block)
        self.main = nn.Sequential(*conv)

    def forward(self, input):
        return self.main(input)
