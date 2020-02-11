import math
import os

import torch
import numpy as np

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
workers = 12

# Batch size during training
batch_size = 128
minibatches = 1


def batchsize_function(epoch, base=batch_size): return base


def minibatch_function(epoch, base=minibatches): return (epoch+1)*base


images = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
end_layer = 1
start_layer = 2
mean_window = 16
factorized_kernel_size = 3
default_kernel_size = 3
# Number of channels in the training images. For color images this is 3
nc = 3
in_size = 2

factor = 2

d_hinge = True
g_hinge = True
g_stride = 2
d_stride = 2
layers = int(math.log(image_size, 2))

factorize = False
use_batch_norm = True
EPSILON = 1e-6
blocks = 0
ROOTTANH_GROWTH = 4
ngf = factor ** int(math.log(image_size, g_stride)) * 16
ndf = factor ** int(math.log(image_size, d_stride)) * 4

min_inception_features = np.uint(-1)  # Never use inception
min_attention_size = 8
attention_every_nth_layer = 2

# Size of z latent vector (i.e. size of generator input)
nz = ngf
# X-size of input.
nx = 2
in_stride = 2
out_x = nx * in_stride
in_size = 2

diters = 1
main_n = 2 ** 10
print_every_nth_batch = max(1, main_n // max(batch_size, 64))
image_intervall = max(1, 16 * main_n / batch_size)
if USE_TPU:
    batch_size = batch_size // 8

glr = 5e-5
dlr = 2e-4
beta1 = 0
beta2 = 0.9

if USE_TPU:
    glr *= 8
    dlr *= 8

transpose_kernel_size = 6

out_field = 5
factor_func = lambda x: 2 ** x

if USE_GPU:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
