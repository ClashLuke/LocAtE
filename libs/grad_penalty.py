def penalty(d_true, aug_data, dis, device, gamma=100):
    return gamma * (d_true.mean() - dis(aug_data.to(device)).view(-1).mean()) ** 2
