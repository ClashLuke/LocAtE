import collections
from functools import reduce
import operator


import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from .config import *
from .nadam import Nadam


def flatten(input_list):
    while input_list:
        while input_list and isinstance(input_list[0], collections.Iterable):
            input_list[0:1] = input_list[0]
        if input_list:
            yield input_list.pop(0)

def kernel_tuple(k, s):
    return tuple([max(k, s)] * 2)


def get_feature_list(strides, invert, features):
    r = range(len(strides) - 1, -1, -1) if invert else range(len(strides))
    feature_list = [features(i) for i in r]
    return feature_list


def conv_pad_tuple(k, _, dim=2):
    return tuple([k // 2] * dim)


def transpose_pad_tuple(k, s, dim=2):
    return tuple([max(conv_pad_tuple(k, s)[0] - s // 2, 0)] * dim)


def view_expand(out_size, tensor):
    if tensor is not None:
        return tensor.view(tensor.size(0), -1, *[1] * (len(tensor.size()) - 2)).expand(out_size)
    return None


def mul_split_add(input_0, input_1, merge_conv):
    features = input_0.size(1)
    input_1 = input_0 + input_1
    factor = merge_conv(torch.cat([input_0, input_1], dim=1))
    factor_0, factor_1 = factor.split(features, 1)
    return input_0 * factor_0 + input_1 * factor_1


def prepare_plot():
    plt.clf()
    #    plt.figure(8, 8)
    plt.axis('off')


def plot_hist(in_list, filename):
    plt.clf()
    plt.plot(in_list)
    plt.savefig(filename)


n = 0


def plot_images(images):
    global n
    prepare_plot()
    dat = np.transpose(vutils.make_grid(images, padding=2, normalize=True).cpu().numpy(), (1, 2, 0))
    plt.imsave(f'{n}.png', np.transpose(vutils.make_grid(images, padding=2, normalize=True).cpu().numpy(), (1, 2, 0)))
    n += 1


def _get_dataset(dataroot, tfs):
    return dset.ImageFolder(root=dataroot, transform=tfs)


def get_dataset(dataroot, image_size, jitter=0.2, min_crop_part=0.75):
    base_tfs = transforms.Compose([transforms.Resize(image_size * 2),
                                   transforms.RandomResizedCrop(image_size, (min_crop_part, 1), (1, 1)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    tfs = transforms.Compose([transforms.Resize(image_size * 2),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(*([jitter] * 3)),
                              transforms.RandomResizedCrop(image_size, (min_crop_part, 1), (1, 1)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ])
    return _get_dataset(dataroot, tfs), _get_dataset(dataroot, base_tfs)


def get_dataloader(dataset, batch_size, workers=4):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=True, num_workers=workers)


def init(m: object):
    if "norm" not in m.__class__.__name__.lower():
        try:
            torch.nn.init.orthogonal_(m.weight.data)
        except AttributeError:
            pass
    else:
        try:
            torch.nn.init.uniform_(m.weight.data, 0.998, 1.002)
        except AttributeError:
            pass
    try:
        torch.nn.init.constant_(m.bias.data, 0)
    except AttributeError:
        pass


def hinge(x):
    return (1 - x).clamp(min=0)


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


model_parameters = lambda net: filter(lambda p: p.requires_grad, net.parameters())
prod_sum = lambda x: sum([np.prod(p.size()) for p in x])
parameter_count = lambda net: prod_sum(model_parameters(net))


def get_model(model, lr, device):
    model = model.to(device)
    model.apply(init)
    opt = Nadam(model.parameters(), lr=lr, betas=(beta1, beta2))
    return model, opt
