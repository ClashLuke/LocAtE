#!/usr/bin/env python
# coding: utf-8

import os
import shutil
# In[1]:
import time
from datetime import timedelta
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
import wget  # Using pip, condas wget doesnt work
from torch.backends import cudnn

from libs import (DATAROOT, DEVICE, DITERS, DLR, Discriminator, GLR, G_HINGE, Generator,
                  IMAGES, IMAGE_SIZE, MAIN_N, MEAN_WINDOW, MINIBATCHES, OUTPUT_FOLDER,
                  OVERFIT, USE_GPU, WORKERS, batchsize_function, get_dataloader,
                  get_dataset, get_model, hinge, minibatch_function, parameter_count,
                  penalty, plot_hist, plot_images)

cudnn.enabled = True
cudnn.benchmark = True

if os.path.isdir('./images/img_align_celeba'):
    print("CelebA already on machine.")
else:
    wget.download("https://archive.org/download/celeba/Img/img_align_celeba.zip")
    try:
        os.mkdir("images")
    except FileExistsError:
        pass
    with ZipFile("img_align_celeba.zip", 'r') as zf:
        zf.extractall("images")
torch.cuda.empty_cache()

# We can use an image folder dataset the way we have it setup.
# Create the dataset
AUGMENTED_DATASET, DATASET = get_dataset(DATAROOT, IMAGE_SIZE)
# Create the dataloader
print(f"Images: {len(get_dataloader(DATASET, 1))}")
IMAGE_COUNT = batchsize_function(0) if OVERFIT else 64
REAL_BATCH, _ = next(iter(get_dataloader(DATASET, IMAGE_COUNT)))
plot_images(REAL_BATCH)
AUGMENTED_BATCH, _ = next(iter(get_dataloader(AUGMENTED_DATASET, IMAGE_COUNT)))
plot_images(AUGMENTED_BATCH)

GEN, GEN_OPTIM = get_model(Generator(), GLR, DEVICE)
DIS, DIS_OPTIM = get_model(Discriminator(), DLR, DEVICE)

G_IN = GEN.g_in

FIXED_NOISE = torch.randn(IMAGES, G_IN, device=DEVICE)

IMG_LIST = []
G_LOSSES = []
D_LOSSES = []

try:
    shutil.rmtree(OUTPUT_FOLDER)
except OSError:
    pass

try:
    os.mkdir(OUTPUT_FOLDER)
except FileExistsError:
    pass

print(DEVICE)

# In[14]:


print(GEN)
print(f'Parameters: {parameter_count(GEN)}')

# In[15]:


print(DIS)
print(f'Parameters: {parameter_count(DIS)}')

torch.autograd.set_detect_anomaly(
        False)  # Advanced debugging at the cost of more errors
print("Starting Training LÖÖPS...")

try:
    os.mkdir('error')
except FileExistsError:
    pass


def dataloader_function(batch):
    while True:
        yield batch, None


def train(*args, **kwargs):
    ditr = DITERS
    gen = GEN
    dis = DIS
    fnoise = FIXED_NOISE
    epoch = -1
    while True:
        dhist = []
        ghist = []
        epoch = 0 if OVERFIT else epoch + 1
        batch = batchsize_function(epoch)
        subepochs = batchsize_function(epoch, 1)
        miniter = minibatch_function(epoch)
        subepochs *= minibatch_function(epoch, 1)
        subepochs *= subepochs
        print_every_nth_batch = max(1, MAIN_N // max(batch, 64))
        image_intervall = max(1, 16 * MAIN_N // batch)
        try:
            os.mkdir(f'{OUTPUT_FOLDER}/{epoch + 1}')
        except FileExistsError:
            pass

        dataloader = torch.utils.data.DataLoader(DATASET, batch_size=batch,
                                                 shuffle=True, num_workers=WORKERS,
                                                 drop_last=True)
        augmented_dataloader = torch.utils.data.DataLoader(AUGMENTED_DATASET,
                                                           batch_size=batch,
                                                           shuffle=True,
                                                           num_workers=WORKERS,
                                                           drop_last=True)

        batches = len(dataloader)
        batch_len = len(str(batches))
        sub_len = len(str(subepochs))
        if OVERFIT:
            dataloader = dataloader_function(REAL_BATCH)
            augmented_dataloader = dataloader_function(AUGMENTED_BATCH)

        for sub in range(subepochs):
            start_time = time.time()
            for i, ((data, _), (aug_data, _)) in enumerate(
                    zip(dataloader, augmented_dataloader), 1):
                noise = torch.randn(batch, G_IN, device=DEVICE)
                if USE_GPU:
                    torch.cuda.empty_cache()
                data = data.to(DEVICE)
                generated = gen(noise).detach()

                dis.zero_grad()

                true_error = hinge(dis(data).view(-1)).mean()
                gen_error = hinge(-dis(generated).view(-1)).mean()
                penalty_error = penalty(true_error, aug_data, dis, DEVICE)
                d_error = true_error + gen_error
                (d_error + penalty_error).backward()

                if i % miniter == 0:
                    DIS_OPTIM.step()
                    if (i // miniter) % ditr == 0:
                        dis.requires_grad_(False)
                        for _ in range(MINIBATCHES):
                            gen.zero_grad()
                            g_error = dis(gen(noise)).view(-1)
                            if G_HINGE:
                                g_error = hinge(g_error).mean()
                            else:
                                g_error = g_error.mean()
                            g_error.backward()

                        GEN_OPTIM.step()
                        dis.requires_grad_(True)

                    if i % print_every_nth_batch == 0:
                        i = i
                        cdiff_i = i / (time.time() - start_time)
                        try:
                            eta = str(timedelta(seconds=int((batches - i) / cdiff_i)))
                        except ZeroDivisionError:
                            eta = "Unknown"
                        try:
                            drror = d_error.item() / 2
                            grror = g_error.item()
                        except UnboundLocalError:
                            continue
                        print(
                                f'\r[{epoch + 1}][{sub + 1}/{subepochs}]['
                                f'{i:{batch_len}d}/{batches}] | Rate: '
                                f'{cdiff_i * batch:.2f} Img/s - {cdiff_i:.2f} Upd/s | '
                                + f'D:{drror:9.4f} - G:{grror:9.4f}| ETA: {eta}',
                                end='', flush=True)
                        dhist.append(drror)
                        ghist.append(grror)
                    if i % image_intervall == 0:
                        gen.eval()
                        if USE_GPU:
                            torch.cuda.empty_cache()
                        with torch.no_grad():
                            fake = gen(fnoise).detach().cpu()
                        if USE_GPU:
                            torch.cuda.empty_cache()
                        gen.train()
                        plt.imsave(
                                f'{OUTPUT_FOLDER}/{epoch + 1}/{sub + 1:0{sub_len}d}-'
                                f'{i:0{batch_len}d}.png',
                                arr=np.transpose(vutils.make_grid(fake, padding=8,
                                                                  normalize=True
                                                                  ).numpy(),
                                                 (1, 2, 0)))
            print('')
            if USE_GPU:
                torch.cuda.empty_cache()
            gen.eval()
            if USE_GPU:
                torch.cuda.empty_cache()
            with torch.no_grad():
                fake = gen(fnoise).detach().cpu()
            if USE_GPU:
                torch.cuda.empty_cache()
            gen.train()
            plt.imsave(f'{OUTPUT_FOLDER}/{epoch + 1}/{sub + 1:0{sub_len}d}-END.png',
                       arr=np.transpose(
                               vutils.make_grid(fake, padding=8,
                                                normalize=True).numpy(),
                               (1, 2, 0)))
        div = ((MEAN_WINDOW ** 2 - MEAN_WINDOW) / 2)
        ma_dhist = [sum(dhist[i + j - 1] * j for j in range(1, MEAN_WINDOW + 1)) / div
                    for i in
                    range(len(dhist) - MEAN_WINDOW)]
        ma_ghist = [sum(ghist[i + j - 1] * j for j in range(1, MEAN_WINDOW + 1)) / div
                    for i in
                    range(len(ghist) - MEAN_WINDOW)]
        plot_hist(ma_dhist, f'error/{epoch + 1}-d.svg')
        plot_hist(ma_ghist, f'error/{epoch + 1}-g.svg')
        torch.save(GEN.state_dict(), 'netG.torch')
        torch.save(DIS.state_dict(), 'netD.torch')


train()
