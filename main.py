#!/usr/bin/env python
# coding: utf-8

# In[1]:
import time
from datetime import timedelta

from torch.backends import cudnn

from libs import *
from libs.config import *

cudnn.enabled = True
cudnn.benchmark = True

os.system(
        'cd ./images/img_align_celeba && echo "CelebA already on machine." || (wget --no-check-certificate "https://archive.org/download/celeba/Img/img_align_celeba.zip" && mkdir images ; unzip -qq "img_align_celeba.zip" -d "images/")')

# In[2]:


# In[3]:


torch.cuda.empty_cache()

# We can use an image folder dataset the way we have it setup.
# Create the dataset
augmented_dataset, dataset = get_dataset(dataroot, image_size)
# Create the dataloader
print(f"Images: {len(get_dataloader(dataset, 1))}")
real_batch, _ = next(iter(get_dataloader(dataset, 64)))
plot_images(real_batch)
augmented_batch, _ = next(iter(get_dataloader(augmented_dataset, 64)))
plot_images(augmented_batch)

### ################## ###
### INITIALISE GLOBALS ###
### ################## ###

netG, optimizerG = get_model(Generator, glr, device)
netD, optimizerD = get_model(Discriminator, dlr, device)

g_in = netG.g_in

fixed_noise = torch.randn(images, g_in, device=device)

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

print(device)

# In[14]:


print(netG)
print(f'Parameters: {parameter_count(netG)}')

# In[15]:


print(netD)
print(f'Parameters: {parameter_count(netD)}')

# In[16]:


# In[ ]:


# First iteration takes 5~10 minutes

torch.autograd.set_detect_anomaly(True)  # Advanced debugging at the cost of more errors
print("Starting Training LÖÖPS...")

try:
    os.mkdir(f'error')
except:
    pass

### ############# ###
### TRAINING LOOP ###
### ############# ###

real_tensor = torch.full((1,), 1, device=device)
fake_tensor = torch.full((1,), -1, device=device)


def train(*args, **kwargs):
    ditr = diters
    global device
    global optimizerG
    global optimizerD
    gen = netG
    dis = netD
    fnoise = fixed_noise
    epoch = -1
    while True:
        dhist = []
        ghist = []
        epoch += 1
        batch = batchsize_function(epoch)
        subepochs = batchsize_function(epoch, 1)
        miniter = minibatch_function(epoch)
        subepochs *= minibatch_function(epoch,1)
        subepochs *= subepochs
        print_every_nth_batch = max(1, main_n // max(batch, 64))
        image_intervall = max(1, 16 * main_n // batch)
        try:
            os.mkdir(f'{OUTPUT_FOLDER}/{epoch + 1}')
        except FileExistsError:
            pass

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch,
                                                 shuffle=True, num_workers=workers,
                                                 drop_last=True)
        augmented_dataloader = torch.utils.data.DataLoader(augmented_dataset, batch_size=batch,
                                                           shuffle=True, num_workers=workers,
                                                           drop_last=True)

        loader = dataloader
        batches = len(dataloader)
        batch_len = len(str(batches))
        sub_len = len(str(subepochs))
        for sub in range(subepochs):
            start_time = time.time()
            for i, ((data, _), (aug_data, _)) in enumerate(zip(loader, augmented_dataloader), 1):
                noise = torch.randn(batch, g_in, device=device)
                if USE_GPU:
                    torch.cuda.empty_cache()
                data = data.to(device)
                generated = gen(noise).detach()

                dis.zero_grad()
                d_true = dis(data).view(-1)
                d_gen = -dis(generated).view(-1)
                if d_hinge:
                  d_error = hinge(d_true) + hinge(d_gen)
                else:
                  d_error = d_true + d_gen
                d_error = d_error.mean()
                (d_error + penalty(d_true, aug_data, dis, device)).backward()

                if i % miniter == 0:
                    optimizerD.step()
                    if (i // miniter) % ditr == 0:
                        dis.requires_grad_(False)
                        for _ in range(minibatches):
                            gen.zero_grad()
                            g_error = dis(gen(noise)).view(-1)
                            if g_hinge:
                              g_error = hinge(g_error).mean()
                            else:
                              g_error = g_error.mean()
                            g_error.backward()

                        optimizerG.step()
                        dis.requires_grad_(True)

                    if i % print_every_nth_batch == 0:
                        i = i
                        cdiff_i = i / (time.time() - start_time)
                        try:
                            eta = str(timedelta(seconds=int((batches - i) / cdiff_i)))
                        except:
                            eta = "Unknown"
                        try:
                            drror = d_error.item() / 2
                            grror = g_error.item()
                        except UnboundLocalError:
                            continue
                        print(
                                f'\r[{epoch + 1}][{sub + 1}/{subepochs}][{i:{batch_len}d}/{batches}] | Rate: {cdiff_i * batch:.2f} Img/s - {cdiff_i:.2f} Upd/s | '
                                + f'D:{drror:9.4f} - G:{grror:9.4f}| ETA: {eta}', end='', flush=True)
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
                        plt.imsave(f'{OUTPUT_FOLDER}/{epoch + 1}/{sub + 1:0{sub_len}d}-{i:0{batch_len}d}.png',
                                   arr=np.transpose(vutils.make_grid(fake, padding=8, normalize=True).numpy(),
                                                    (1, 2, 0)))
            print('')
            if USE_GPU:
                torch.cuda.empty_cache()
        div = ((mean_window ** 2 - mean_window) / 2)
        ma_dhist = [sum(dhist[i + j - 1] * j for j in range(1, mean_window + 1)) / div for i in
                    range(len(dhist) - mean_window)]
        ma_ghist = [sum(ghist[i + j - 1] * j for j in range(1, mean_window + 1)) / div for i in
                    range(len(ghist) - mean_window)]
        plot_hist(ma_dhist, f'error/{epoch + 1}-d.svg')
        plot_hist(ma_ghist, f'error/{epoch + 1}-g.svg')
        torch.save(netG.state_dict(), 'netG.torch')
        torch.save(netD.state_dict(), 'netD.torch')


train()
