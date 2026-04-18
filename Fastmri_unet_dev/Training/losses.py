'''
THis file is used for implement losses
'''
import torch
from torch import nn
from torch.nn import functional as F

import config_file


def create_window(window_size, num_channel, device):
    '''
    Create convolution kernel
    Args:
        window_size: convolution kernel size
        num_channel: image number of channels
    Returns:
        window: convolution kernel
    '''

    window = torch.ones((num_channel, 1, window_size, window_size))
    window /= window_size ** 2
    return window.to(device)

def _ssim(img1, img2, window, window_size=config_file.WINDOW_SIZE, k1=0.01, k2=0.03, data_range=1.0):
    '''
    Compute Structural Similarity Index Metric (SSIM) value
    Args:
        img1 (torch tensor): 2D image (H,W)
        img2 (torch tensor): 2D image (H,W)
        window (torch tensor): 2D convolution kernel
        window_size: convolution kernel size
        k1: scaler constant
        k2: scaler constant
        data_range: value range of input images, in our case uses 1.0

    Returns:
        pixel-wise SSIM
    '''
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=img2.shape[1])

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=img1.shape[1]) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map

def ssim_loss(input, target, window_size=11, size_average=True):
    '''
    Compute SSIM value with a uniform convolution kernel
    Args:
        input (torch tensor): 2D image (N,H,W)
        target (torch tensor): 2D image (N,H,W)
        window_size: convolution kernel size
        size_average (bool): If size_average is True, take average of ssim values of all images

    Returns:
        ssim: SSIM values over the whole batch of images
    '''
    # Create window for convolution
    device = input.device  # Get the device from the input tensor
    window = create_window(window_size, input.size(1), device)  # Pass the device to the function
    ssim_map = _ssim(input, target, window, window_size)

    if size_average:
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map.sum()

    return ssim

def mse_loss(input, target):
    '''
    Compute MSE loss with
    Args:
        input:
        target:

    Returns:
    '''
    mse_loss = nn.MSELoss()
    return mse_loss(input, target)





