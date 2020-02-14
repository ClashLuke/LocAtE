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
image_size = 32
end_layer = 1
start_layer = 0
mean_window = 16
factorized_kernel_size = 3
default_kernel_size = 3
# Number of channels in the training images. For color images this is 3
factor = 2

d_hinge = True
g_hinge = True
g_stride = 2
d_stride = 2
layers = int(math.log(image_size, 2))

factorize = False
ROOTTANH_GROWTH = 4
ngf = factor ** int(math.log(image_size, g_stride)) * 16
ndf = factor ** int(math.log(image_size, d_stride)) * 4
BOTTLENECK = 4
min_inception_features = np.uint(-1)  # Never use inception
min_attention_size = 8
attention_every_nth_layer = 2
input_vector_z = image_size
diters = 1
main_n = 2 ** 10

glr = 5e-4
dlr = 2e-3
beta1 = 0
beta2 = 0.9

if USE_GPU:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
