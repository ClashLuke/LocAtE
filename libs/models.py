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

nz = quadnorm(nz)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        global g_in
        clayers = layers - int(math.log(out_x)) - 1
        stride_count_factor = g_stride // 2
        strides = [g_stride] * (clayers // stride_count_factor) + [2] * (clayers % stride_count_factor)
        global G_LAYERS
        global g_features
        G_LAYERS = len(strides)

        in_f = nz
        g_features = GFeatures(0, clayers)
        out_f = g_features(clayers)
        feature_list = get_feature_list([1]+strides, True, g_features)

        self.input_block = nn.Sequential(*[ResModule(lambda x: x,
                                                     nn.Sequential(*FactorizedConv(in_f*(not(i))+out_f*bool(i), out_f, nx + 1, 1, nx // 2)))
                                           for i in range(start_layer)])
        self.conv_block = BlockBlock(len(strides), nx, feature_list, strides, True, True)
        self.out_conv = nn.Sequential(*FactorizedConv(self.conv_block.out_features, 3,
                                                      kernel_size=kernel_tuple(transpose_kernel_size, 2),
                                                      stride=2,
                                                      padding=transpose_pad_tuple(transpose_kernel_size, 2),
                                                      end_nonlinear=False,
                                                      nonlinearity=False,
                                                      transpose=True),
                                      nn.Tanh())

        self.g_in = feature_list[0]

        self.input = torch.randn(1, nz, nx, nx, device=device)

    def forward(self, input):
        expanded_input = self.input.expand(input.size(0), -1, -1, -1)
        conv_out = self.input_block(expanded_input)
        conv_out = self.conv_block(conv_out, input)
        conv_out = self.out_conv(conv_out)
        return conv_out


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
                               nn.Sequential(*FactorizedConv(3, d_features(0), (5, 5), 2, 2, batch_norm=True, factorize=False)))
        block_block = BlockBlock(len(strides), image_size // 2, feature_list, strides, False)
        conv = [ResModule(lambda x: x,
                          nn.Sequential(*FactorizedConv(block_block.out_features, block_block.out_features, 1)))
                for i in range(end_layer - 1)]
        conv.append(nn.Sequential(*FactorizedConv(block_block.out_features, 1, 1,
                                                  end_nonlinear=False, nonlinearity=False)))
        conv.insert(0, cat_module)
        conv.insert(1, block_block)
        self.main = nn.Sequential(*conv)

    def forward(self, input):
        return self.main(input)

