import math

import numpy as np
import torch

# Set random seed for reproducibility
print("Random Seed: ", 999)
torch.manual_seed(999)

OUTPUT_FOLDER = 'gan_outputs'
GPU_COUNT = 1
USE_GPU = torch.cuda.is_available() and GPU_COUNT > 0
# Root directory for dataset
DATAROOT = "images/"

# Number of workers for dataloader
WORKERS = 12

# Batch size during training
BATCH_SIZE = 128
MINIBATCHES = 1


def batchsize_function(epoch, base=BATCH_SIZE): return int((epoch + 1) ** 0.5 * base)


def minibatch_function(epoch, base=MINIBATCHES): return int((epoch + 1) ** 0.5 * base)


IMAGES = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
IMAGE_SIZE = 64
END_LAYER = 1
START_LAYER = 0
MEAN_WINDOW = 16
FACTORIZED_KERNEL_SIZE = 3
DEFAULT_KERNEL_SIZE = 3
# Number of channels in the training images. For color images this is 3
FACTOR = 2

D_HINGE = True
G_HINGE = True
G_STRIDE = 2
D_STRIDE = 2
LAYERS = int(math.log(IMAGE_SIZE, 2))

FACTORIZE = False
SEPARABLE = False
OVERFIT = False
ROOTTANH_GROWTH = 4
GEN_FEATURES = FACTOR ** int(math.log(IMAGE_SIZE, G_STRIDE)) * 16
DIS_FEATURES = FACTOR ** int(math.log(IMAGE_SIZE, D_STRIDE)) * 4
BOTTLENECK = 4
MIN_INCEPTION_FEATURES = np.uint(-1)  # Never use inception
MIN_ATTENTION_SIZE = 8
ATTENTION_EVERY_NTH_LAYER = 2
INPUT_VECTOR_Z = IMAGE_SIZE
DITERS = 1
MAIN_N = 2 ** 10

GLR = 5e-5
DLR = 2e-4
BETA_1 = 0.5
BETA_2 = 0.9

if USE_GPU:
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')
