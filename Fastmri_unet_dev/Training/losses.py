'''
THis file is used for implement losses
'''
import torch
from torch import nn
from torch.nn import functional as F


def create_window(window_size, num_channel):
    '''
    Create convolution kernel
    Args:
        window_size: convolution kernel size
        num_channel: image number of channels
    Returns:
        window: convolution kernel
    '''

    window = torch.ones(num_channel, 1, window_size, window_size)
    window /= (window_size ** 2)        # uniform window

    return window
def _ssim(img1, img2, window, window_size=11, k1=0.01, k2=0.03, data_range=1.0):
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
    C1 = (k1 * data_range)**2
    C2 = (k2 * data_range)**2

    mu1 = F.con2d(img1, window, padding=window_size//2)
    mu2 = F.con2d(img2, window, padding=window_size//2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu_cross = mu1 * mu2

    sig_cross = F.con2d(img1 * img2, window, padding=window_size//2) - mu_cross
    sig1_sq = F.con2d(img1 * img1, window, padding=window_size//2) - mu1_sq
    sig2_sq = F.con2d(img2 * img2, window, padding=window_size//2) - mu2_sq

    Luminance = (2 * mu_cross + C1) / (mu1_sq + mu2_sq + C1)
    Contrast = (2 * sig_cross + C2) / (sig1_sq + sig2_sq + C2)

    ssim = Luminance * Contrast

    return ssim

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
    window = create_window(window_size=window_size, num_channel=1)
    # window = window.to(current_device)

    ssim_val = _ssim(input, target, window)

    if size_average:
        ssim = ssim_val.mean()
    else:
        ssim = ssim_val.sum()

    return ssim

def mse_loss(input, target):
    '''
    Compute MSE loss with
    Args:
        input:
        target:

    Returns:

    '''

    return F.MSELoss(input, target)





