#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

get_ipython = lambda: os

get_ipython().system(
    'cd ./images/img_align_celeba && echo "CelebA already on machine." || (wget --no-check-certificate "https://archive.org/download/celeba/Img/img_align_celeba.zip" && mkdir images ; unzip -qq "img_align_celeba.zip" -d "images/")')
# !cd ./images/train_64x64 && echo "ImageNet already on machine." || (wget "http://www.image-net.org/small/train_64x64.tar" && mkdir images ; tar -xf "train_64x64.tar" -C "images/")

get_ipython().system('pip install torchviz')

# In[2]:


if os.environ.get('COLAB_TPU_ADDR', False):
    os.environ['TRIM_GRAPH_SIZE'] = "500000"
    os.environ['TRIM_GRAPH_CHECK_FREQUENCY'] = "20000"

    if not os.path.exists('/content/torchvision-1.15-cp36-cp36m-linux_x86_64.whl'):
        import collections
        from datetime import datetime, timedelta
        import requests
        import threading

        _VersionConfig = collections.namedtuple('_VersionConfig', 'wheels,server')
        VERSION = "xrt==1.15.0"  # @param ["xrt==1.15.0", "torch_xla==nightly"]
        CONFIG = {
            'xrt==1.15.0': _VersionConfig('1.15', '1.15.0'),
            'torch_xla==nightly': _VersionConfig('nightly', 'XRT-dev{}'.format(
                (datetime.today() - timedelta(1)).strftime('%Y%m%d'))),
        }[VERSION]
        DIST_BUCKET = 'gs://tpu-pytorch/wheels'
        TORCH_WHEEL = 'torch-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)
        TORCH_XLA_WHEEL = 'torch_xla-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)
        TORCHVISION_WHEEL = 'torchvision-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)


        # Update TPU XRT version
        def update_server_xrt():
            print('Updating server-side XRT to {} ...'.format(CONFIG.server))
            url = 'http://{TPU_ADDRESS}:8475/requestversion/{XRT_VERSION}'.format(
                TPU_ADDRESS=os.environ['COLAB_TPU_ADDR'].split(':')[0],
                XRT_VERSION=CONFIG.server,
            )
            print('Done updating server-side XRT: {}'.format(requests.post(url)))


        update = threading.Thread(target=update_server_xrt)
        update.start()

        # Install Colab TPU compat PyTorch/TPU wheels and dependencies
        os.system("""pip uninstall -y torch torchvision ; gsutil cp "$DIST_BUCKET/$TORCH_WHEEL" . ; """
                  + """gsutil cp "$DIST_BUCKET/$TORCH_XLA_WHEEL" . ; gsutil cp "$DIST_BUCKET/$TORCHVISION_WHEEL" . ;"""
                  + """pip install "$TORCH_WHEEL";pip install "$TORCH_XLA_WHEEL";pip install "$TORCHVISION_WHEEL";"""
                  + """sudo apt-get install libomp5""")
        update.join()

# In[3]:


import os
import time
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import math

if os.environ.get('COLAB_TPU_ADDR', False):
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp

torch.cuda.empty_cache()

# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
torch.manual_seed(manualSeed)

OUTPUT_FOLDER = 'gan_outputs'
ngpu = 1
USE_TPU = os.environ.get('COLAB_TPU_ADDR', False)
USE_GPU = torch.cuda.is_available() and ngpu > 0
# Root directory for dataset
dataroot = "images/"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128
batchsize_increase = 1.05
images = min(batch_size, 64)  # 64++

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3
in_size = 4
layers = int(math.log(image_size, 2))

factor = 2

factorize = False
cat_out = True

EPSILON = 1e-6
blocks = 0

ngf = image_size * 2 ** (layers // 2)
ndf = image_size * 2 ** (layers // 2 - 1)

# Size of z latent vector (i.e. size of generator input)
nz = ngf
# X-size of input.
nx = 2
in_stride = 2
out_x = nx * in_stride
in_size = 4

# Number of training epochs
num_epochs = 6

# Learning rate for optimizers
diters = 2
main_n = 2 ** 10
print_every_nth_batch = max(1, main_n // max(batch_size, 64))
image_intervall = max(1, 16 * main_n / batch_size)
if USE_TPU:
    batch_size = batch_size // 8

glr = 1 * 10 ** -4
dlr = 2 * 10 ** -3

if USE_TPU:
    glr *= 8
    dlr *= 8

transpose_kernel_size = 6

# Number of GPUs available. Use 0 for CPU mode.
out_field = 5
factor_func = lambda x: 2 ** x

min_crop_part = 0.75
jitter = 0.25
# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size * 2),
                               transforms.RandomHorizontalFlip(),
                               transforms.ColorJitter(*([jitter] * 3)),
                               transforms.RandomResizedCrop(image_size, (min_crop_part, 1), (1, 1)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=True, num_workers=workers)

print(f"Images: {len(dataloader)}")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
if USE_GPU:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device), padding=2, normalize=True).cpu(), (1, 2, 0)))

# In[4]:


import math
from torch.optim import Optimizer


class AdaBound(Optimizer):
    """Implements AdaBound algorithm.
    It has been proposed in `Adaptive Gradient Methods with Dynamic Bound of Learning Rate`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): Adam learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        final_lr (float, optional): final (SGD) learning rate (default: 0.1)
        gamma (float, optional): convergence speed of the bound functions (default: 1e-3)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsbound (boolean, optional): whether to use the AMSBound variant of this algorithm
    .. Adaptive Gradient Methods with Dynamic Bound of Learning Rate:
        https://openreview.net/forum?id=Bkg3g2R9FX
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        defaults = dict(lr=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        weight_decay=weight_decay, amsbound=amsbound)
        super(AdaBound, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))

    def __setstate__(self, state):
        super(AdaBound, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')
                amsbound = group['amsbound']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsbound:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Applies bounds on actual learning rate
                # lr_scheduler cannot affect final_lr, this is a workaround to apply lr decay
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                p.data.add_(-step_size)

        return loss


# In[5]:


import torch

from torch.autograd import Variable
from torch import nn
from torch.nn import Parameter


def l2normalize(v, eps=EPSILON):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


# In[6]:


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.slope = torch.nn.Parameter(torch.ones(1))

    def forward(self, input):
        return input.mul(torch.sigmoid(torch.mul(input, self.slope.data)).add(EPSILON))


class LogActivation(nn.Module):
    def __init__(self):
        self.a = torch.nn.Parameter(torch.zeros(1))
        self.b = torch.nn.Parameter(torch.ones(1))
        self.c = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input):
        (c + x) * log2(abs(a) + 1 + (b + 1 + EPSILON) ^ x)
        return self.c.data.add(input).mul(
            torch.log2(torch.abs(self.a.data).add(1).add(self.b.data.add(1 + EPSILON).pow(input))))


# nlinear = nn.ReLU
nlinear = Swish
norm = nn.BatchNorm2d


# In[7]:


def view_expand(out_size, tensor):
    if tensor is not None:
        return tensor.view(tensor.size(0), -1, 1, 1).expand(out_size)
    return None


def mul_split_add(input_0, input_1, merge_conv):
    features = input_0.size(1)
    input_1 = input_0 + input_1
    factor = merge_conv(torch.cat([input_0, input_1], dim=1))
    factor_0, factor_1 = factor.split(features, 1)
    return input_0 * factor_0 + input_1 * factor_1


class Expand(nn.Module):
    def __init__(self, *target_size):
        super(Expand, self).__init__()
        self.target_size = target_size

    def forward(self, input):
        return view_expand(self.target_size, input)


class Concatenate(nn.Module):
    def __init__(self, dim):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, *input):
        return torch.cat(input, self.dim)


class Sum(nn.Module):
    def __init__(self, target_features):
        super(Sum, self).__init__()
        self.target_features = target_features
        self.norm = norm(target_features)
        self.nlin = nlinear()

    def forward(self, input):
        return self.nlin(self.norm(torch.stack(input.split(self.target_features, dim=1), dim=0).sum(dim=0)))


# In[8]:


def FactorizedConv(in_features, out_features, kernel_size=1, stride=1, padding=(0, 0),
                   nonlinearity=True, end_nonlinear=True, bias=False, transpose=False):
    layers = []
    if transpose:
        conv = nn.ConvTranspose2d
    else:
        conv = nn.Conv2d
    if isinstance(kernel_size, int):
        if kernel_size == 1:
            layers.append(conv(in_features, out_features, kernel_size=(1, 1), bias=bias))
        else:
            kernel_size = (kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(kernel_size, tuple):
        if in_features == out_features and factorize:
            layers.append(conv(in_features, in_features, kernel_size=(kernel_size[0], 1),
                               padding=(padding[0], 0), stride=(stride, 1), bias=bias))
            layers.append(conv(in_features, in_features, kernel_size=(1, kernel_size[1]),
                               padding=(0, padding[1]), stride=(1, stride), bias=bias))
        else:
            layers.append(conv(in_features, out_features, kernel_size=kernel_size,
                               padding=padding, stride=stride, bias=bias))
    layers = [SpectralNorm(l) for l in layers]
    if nonlinearity:
        layers.insert(1, nlinear())
    if end_nonlinear:
        layers.append(nlinear())
    if nonlinearity and end_nonlinear and isinstance(layers[-1], nlinear) and isinstance(layers[-2], nlinear):
        del layers[-1]
    return layers


### ################ ###
### INCEPTION LAYERS ###
### ################ ###

kernel_tuple = lambda k, s: tuple([max(k, s)] * 2)
conv_pad_tuple = lambda k, _: tuple([k // 2] * 2)
transpose_pad_tuple = lambda k, s: tuple([max(conv_pad_tuple(k, s)[0] - s // 2, 0)] * 2)


class InceptionBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, transpose,
                 block_nlinear=True):
        super(InceptionBlock, self).__init__()

        transpose = transpose and stride > 1
        kernels = [3, 5]

        if out_features // len(kernels) * len(kernels) == out_features:
            out_features = out_features // len(kernels)
        else:
            kernels = [max(kernels)]
        if transpose:
            kernels = [k - 1 for k in kernels]
            pad_tuple = transpose_pad_tuple
        else:
            pad_tuple = conv_pad_tuple
        c = lambda x: [L for C in [FactorizedConv(in_features, out_features,
                                                  kernel_size=x,
                                                  stride=stride,
                                                  padding=pad_tuple(x, stride),
                                                  nonlinearity=block_nlinear,
                                                  end_nonlinear=block_nlinear,
                                                  transpose=transpose)]
                       for L in C]

        layers = [nn.Sequential(*c(k)) for k in kernels]
        self.layers = layers
        for k, L in zip(kernels, layers):
            setattr(self, f'factorized_conv{k}x{k}', L)

        self.out = lambda x: torch.cat([l(x) for l in layers], dim=1)

    def forward(self, input):
        out = self.out(input)
        return out


# In[9]:


class ResModule(nn.Module):
    def __init__(self, features, residual_module, layer_module):
        super(ResModule, self).__init__()
        self.features = features
        self.residual_module = residual_module
        self.layer_module = layer_module
        self.layer_norm = norm(features)
        self.layer_nlin = nlinear()
        self.merger = SpectralNorm(nn.Conv2d(features * 2, features * 2, kernel_size=1))
        self.output_norm = norm(features)
        self.output_nlin = nlinear()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, input, layer_input=None):
        if layer_input is not None:
            layer_out = self.layer_module(layer_input)
            res = input
        else:
            res, layer_out = self.residual_module(input), self.layer_module(input)
        layer_out = layer_out + res
        layer_out = self.layer_nlin(self.layer_norm(layer_out))
        cat_layer = torch.cat([res, layer_out], dim=1)
        res_factor, layer_factor = self.softmax(torch.stack(self.merger(cat_layer).split(self.features, 1), 0))
        output = res_factor * res + layer_factor * layer_out
        output = self.output_nlin(self.output_norm(output))
        return output


# In[10]:


def Scale(in_features, out_features, stride, transpose):
    reslayers = []
    if in_features != out_features:
        reslayers.extend(FactorizedConv(in_features, out_features, 1,
                                        nonlinearity=True,
                                        end_nonlinear=True))
    if stride > 1:
        if transpose:
            reslayers.append(nn.Upsample(mode='bilinear', scale_factor=stride,
                                         align_corners=False))
        else:
            reslayers.append(nn.AvgPool2d(stride, stride))
    if len(reslayers) > 1:
        return nn.Sequential(*reslayers)
    elif not reslayers:
        return lambda x: x
    else:
        return reslayers[0]


# In[11]:


def Attention(in_size, features):
    return nn.Sequential(SpectralNorm(nn.Conv2d(features, features, kernel_size=(1, in_size))),
                         nlinear(),
                         SpectralNorm(nn.Conv2d(features, features, kernel_size=(in_size, 1))),
                         Expand(-1, features, in_size, in_size)
                         )


# In[12]:


class Block(nn.Module):
    def __init__(self, in_size, in_features, out_features, stride, transpose, cat_out=cat_out):
        super(Block, self).__init__()
        self.scale_layer = Scale(in_features, out_features, stride, transpose)
        inception_layer = InceptionBlock(in_features, out_features, stride, transpose)
        self.out_module_0 = ResModule(out_features, lambda x: x, inception_layer)
        attention_layer = Attention(in_size, out_features)
        self.out_module_1 = ResModule(out_features, lambda x: x, attention_layer)
        if cat_out:
            self.out_module_2 = Concatenate(1)
        else:
            self.out_module_2 = ResModule(out_features, lambda x: x, lambda x: x)

    def forward(self, input):
        scaled = self.scale_layer(input)
        out = self.out_module_0(scaled, input)
        out = self.out_module_1(out, out)
        out = self.out_module_2(scaled, out)
        return out


# In[13]:


### ################# ###
### FACTORIZED LAYERS ###
### ################# ###

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


### ########## ###
### MLP LAYERS ###
### ########## ###

def BaseMLP(in_features, out_features, activation=nn.Tanh(), dropout=False, **kwargs):
    layers = [nn.Linear(in_features, out_features, bias=False)]
    layers.append(activation())
    if dropout:
        layers.append(nn.Dropout)
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


### ######## ###
### NETWORKS ###
### ######## ###

quadnorm = lambda x: x // 4 * 4
D_LAYERS = 0
G_LAYERS = 0

g_features = lambda x: (quadnorm((1 + int(cat_out)) * int(ngf * factor ** ((x + 1) - G_LAYERS))),
                        quadnorm(int(ngf * factor ** ((x) - G_LAYERS))))
d_features = lambda x: (quadnorm((1 + int(cat_out)) * int(ndf * factor ** (((x) - D_LAYERS)))),
                        quadnorm(int(ndf * factor ** (((x + 1) - D_LAYERS)))))

g_in = None


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        global g_in
        clayers = layers - int(math.log(out_x)) - 1
        strides = [2] * (clayers)
        global G_LAYERS
        G_LAYERS = clayers

        conv_layers = MultiInception(2, strides, g_features, transpose=True,
                                     invert=True, sequential=True,
                                     block_nlinear=False,
                                     in_features=nz)
        conv_layers.append(nn.Sequential(*FactorizedConv(g_features(-1)[0], 3,
                                                         kernel_size=kernel_tuple(transpose_kernel_size, 2),
                                                         stride=2,
                                                         padding=transpose_pad_tuple(transpose_kernel_size, 2),
                                                         end_nonlinear=False,
                                                         nonlinearity=False,
                                                         transpose=True)))

        for i, L in enumerate(conv_layers):
            setattr(self, f'transp_{i}', L)
        self.conv_layers = conv_layers

        mlp_std_layers = MultiMLP(clayers, g_features, invert=True, sequential=True, activation=nn.PReLU)
        g_in = g_features(len(strides) - 1)[0]

        for i, L in enumerate(mlp_std_layers):
            setattr(self, f'mlp_std_{i}', L)
        self.mlp_std_layers = mlp_std_layers

        self.out = nn.Tanh()
        self.noise = torch.randn(1, nz, nx, nx, device=device)

    def forward(self, input):
        batch = input.size(0)
        c_out = self.noise.expand(batch, nz, nx, nx)
        m_out = input
        s_out = input

        for c, s in zip(self.conv_layers, self.mlp_std_layers):
            s_out = s(s_out)
            c_out = c(c_out)
            c_out = c_out * view_expand(c_out.size(), s_out)
        c_out = self.conv_layers[-1](c_out)
        return self.out(c_out)


class Discriminator(nn.Module):
    def __init__(self):
        global D_LAYERS
        super(Discriminator, self).__init__()
        clayers = layers - 1
        strides = [4] * (clayers // 2)
        D_LAYERS = len(strides)

        conv = [SpectralNorm(nn.Conv2d(3, d_features(0)[0], (5, 5), padding=2, stride=2, bias=False))]
        conv.extend(
            MultiInception(image_size // 2, features=d_features, strides=strides, sequential=True, block_nlinear=False))
        conv.append(SpectralNorm(nn.Conv2d(d_features(len(strides))[0], 1, (3, 3), padding=1, bias=False)))

        self.main = nn.Sequential(*conv,
                                  nn.Flatten(start_dim=1))

    def forward(self, input):
        return self.main(input).mean(dim=1)


### ################### ###
### INITIALISE NETWORKS ###
### ################### ###

def init(m):
    try:
        torch.nn.init.xavier_uniform_(m.weight.data)
    except AttributeError:
        pass
    except ValueError:
        torch.nn.init.uniform_(m.weight.data, 0.998, 1.002)
    try:
        torch.nn.init.constant_(m.bias.data, 0)
    except AttributeError:
        pass


model_parameters = lambda net: filter(lambda p: p.requires_grad, net.parameters())
prod_sum = lambda x: sum([np.prod(p.size()) for p in x])
parameter_count = lambda net: prod_sum(model_parameters(net))


def get_model(model_class, lr, device):
    model = model_class().to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        model = nn.DataParallel(model, list(range(ngpu)))
    model.apply(init)
    #  opt = torch.optim.RMSprop(model.parameters(), lr=lr, eps=EPSILON)
    #  opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5,0.999), eps=EPSILON)
    opt = AdaBound(model.parameters(), lr=lr, final_lr=lr * 100, betas=(0.5, 0.999), eps=EPSILON)
    return model, opt


### ################## ###
### INITIALISE GLOBALS ###
### ################## ###

netG, optimizerG = get_model(Generator, glr, device)
netD, optimizerD = get_model(Discriminator, dlr, device)

fixed_noise = torch.randn(images, g_in, device=device)

real_label = 0
fake_label = 1

img_list = []
G_losses = []
D_losses = []

### ########################### ###
### INITIALISE FOLDER STRUCTURE ###
### ########################### ###

import shutil

try:
    shutil.rmtree(OUTPUT_FOLDER)
except:
    pass

try:
    os.mkdir(OUTPUT_FOLDER)
except FileExistsError:
    pass

f_n = lambda x, n: str(int(x * 10 ** n) * 10 ** (-n))

print(device)

# In[14]:


print(netG)
print(f'Parameters: {parameter_count(netG)}')

# In[15]:


print(netD)
print(f'Parameters: {parameter_count(netD)}')


# In[16]:


def penalty(data, generated_data, dis, device, gamma=100):
    batch_size = data.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    epsilon = epsilon.expand_as(data)

    interpolation = epsilon * data.data + (1 - epsilon) * generated_data.data
    interpolation = torch.autograd.Variable(interpolation, requires_grad=True)

    if USE_GPU:
        interpolation = interpolation.cuda()

    interpolation_logits = dis(interpolation)
    grad_outputs = torch.ones(interpolation_logits.size())

    if USE_GPU:
        grad_outputs = grad_outputs.cuda()

    gradients = torch.autograd.grad(outputs=interpolation_logits,
                                    inputs=interpolation,
                                    grad_outputs=grad_outputs,
                                    create_graph=True,
                                    retain_graph=True)[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + EPSILON)
    return gamma * ((gradients_norm - 1) ** 2).mean()


# In[ ]:


# First iteration takes 5~10 minutes

torch.autograd.set_detect_anomaly(False)  # Advanced debugging at the cost of more errors
print("Starting Training LÖÖPS...")

### ############# ###
### TRAINING LOOP ###
### ############# ###

dist_hist = []
real_tensor = torch.full((1,), 1, device=device)
fake_tensor = torch.full((1,), -1, device=device)


def train(*args, **kwargs):
    ditr = diters * 64
    batch = batch_size
    if not USE_TPU:
        global device
        global optimizerG
        global optimizerD
        gen = netG
        dis = netD
        fnoise = fixed_noise
    else:
        device = xm.xla_device()
        gen = netG.to(device)
        dis = netD.to(device)
        fnoise = fixed_noise.to(device)
        gen.noise = gen.noise.to(device)
    epoch = -1
    while True:
        epoch += 1
        subepochs = int(batchsize_increase ** epoch)
        batch = batch_size * subepochs
        if USE_TPU:
            print_every_nth_batch = max(1, main_n // max(8 * batch, 64))
            image_intervall = max(1, 2 * main_n / batch)
        else:
            print_every_nth_batch = max(1, main_n // max(batch, 64))
            image_intervall = max(1, 16 * main_n / batch)
        try:
            os.mkdir(f'{OUTPUT_FOLDER}/{epoch + 1}')
        except FileExistsError:
            pass

        if USE_TPU:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                      num_replicas=xm.xrt_world_size(),
                                                                      rank=xm.get_ordinal(),
                                                                      shuffle=True)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch,
                                                     sampler=sampler, num_workers=workers,
                                                     drop_last=True)
            parallel_loader = pl.ParallelLoader(dataloader, [device])
            loader = parallel_loader.per_device_loader(device)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch,
                                                     shuffle=True, num_workers=workers,
                                                     drop_last=True)
            loader = dataloader
        batches = len(dataloader)
        batch_len = len(str(batches))
        sub_len = len(str(subepochs))
        for sub in range(subepochs):
            start_time = time.time()
            for i, (data, _) in enumerate(loader, 1):
                gen.noise = gen.noise.to(device)
                noise = torch.randn(batch, g_in, device=device)
                if USE_GPU:
                    torch.cuda.empty_cache()
                data = data.to(device)
                generated = gen(noise).detach()

                dis.zero_grad()
                d_error = dis(data).view(-1).mean() - dis(generated).view(-1).mean() + penalty(data, generated, dis,
                                                                                               device)
                d_error.backward()

                if USE_TPU:
                    xm.optimizer_step(optimizerD)
                else:
                    optimizerD.step()
                if i % ditr == 0:
                    dis.requires_grad_(False)
                    for p in dis.parameters():  # Reset requires_grad
                        p.requires_grad = False  # To avoid computation
                    gen.zero_grad()
                    dis(gen(noise)).view(-1).mean().backward()

                    if USE_TPU:
                        xm.optimizer_step(optimizerG)
                    else:
                        optimizerG.step()

                    for p in dis.parameters():  # Reset requires_grad
                        p.requires_grad = True  # Reset to default for discriminator above

                    # Output training stats
                if i % print_every_nth_batch == 0 and (not USE_TPU or xm.get_ordinal() == 0):
                    cdiff_i = i / (time.time() - start_time)
                    try:
                        eta = str(timedelta(seconds=int((batches - i) / cdiff_i)))
                    except:
                        eta = "Unknown"
                    if USE_TPU:
                        cdiff_i *= 8
                    dist = -d_error.item()
                    print(
                        f'\r[{epoch + 1}/{num_epochs}][{sub + 1}/{subepochs}][{i:{batch_len}d}/{batches}] | Rate: {cdiff_i * batch:.2f} Img/s - {cdiff_i:.2f} Upd/s | '
                        + f'Dist:{dist:9.4f} | ETA: {eta}', end='', flush=True)
                    dist_hist.append(dist)
                    if not USE_TPU:
                        if i % image_intervall == 0:
                            if USE_GPU:
                                torch.cuda.empty_cache()
                            with torch.no_grad():
                                fake = gen(fnoise).detach().cpu()
                            if USE_GPU:
                                torch.cuda.empty_cache()
                            plt.imsave(f'{OUTPUT_FOLDER}/{epoch + 1}/{sub + 1:0{sub_len}d}-{i:0{batch_len}d}.png',
                                       arr=np.transpose(vutils.make_grid(fake, padding=8, normalize=True).numpy(),
                                                        (1, 2, 0)))
            print('')
            if USE_TPU:
                show_images()
            if USE_GPU:
                torch.cuda.empty_cache()
        batch *= batchsize_increase


if USE_TPU:
    loop = True
    while loop:
        try:
            xmp.spawn(train, nprocs=8, start_method='fork')
        except KeyboardInterrupt:
            loop = False
        except BaseException as exc:
            print('')
            show_images()
else:
    train()

# In[ ]:


plt.clf()
plt.axis("on")
plt.title("Fake-Real Distance")
plt.plot(dist_hist)
plt.show()
plt.clf()

# In[ ]:


torch.save(netG.state_dict(), 'netG.torch')
torch.save(netD.state_dict(), 'netD.torch')

# In[ ]:


if USE_GPU:
    torch.cuda.empty_cache()
with torch.no_grad():
    fake = netG(fixed_noise).detach()
if USE_GPU:
    torch.cuda.empty_cache()

plt.figure(figsize=(8, 8))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(fake.to(device), padding=8, normalize=True).cpu().numpy(), (1, 2, 0)))

# In[ ]:


seconds_per_epoch = 16
with open('list.txt', 'w') as f:
    f.write('')
with open('outputs.txt', 'w') as f:
    f.write('')
framerate = min(30, len(dataloader) / image_intervall / seconds_per_epoch)
for i in range(num_epochs):
    os.system(
        f'rm out{i + 1}.mp4 ; ffmpeg -framerate {framerate} -pattern_type glob -i \'./gan_outputs/{i + 1}/*.png\' -c:v libx264 -pix_fmt yuv420p out{i + 1}.mp4 && echo "file {os.getcwd()}/out{1 + i}.mp4" >> list.txt')
os.system("rm output.mp4 ; ffmpeg -f concat -safe 0 -i list.txt -c copy output.mp4")
