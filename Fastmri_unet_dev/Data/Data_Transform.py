      cimport numpy
import numpy as np
import torch
import  Data.Fastmri_Data_Processing
import torch.nn as nn
import torch.nn.functional as F
import h5py
from fastmri.data.subsample import RandomMaskFunc
import fastmri
from fastmri.data import transforms as T
'''
owner: Fang Zou
'''


def apply_sub_sample(k_space: numpy.array):
    '''
    :param k_space: numpy array with dims(num_slices, w, h)
    :return: k_space_sub_sampled(num_slices, w, h ) , apply zero_ filled mask, to mask out the original k_space
    '''
    #todo
    # s= k_space.shape
    # k_space_sub_sampled = np.zeros(s)
    mask_func = RandomMaskFunc(center_fractions=[0.04], accelerations=[8])
    k_space_sub_sampled, mask, _ = T.apply_mask(k_space, mask_func) 

    return k_space_sub_sampled

def inverse_fft(sub_sample_kspace: numpy.array):
    '''
    
    :param sub_sample_kspace: dim(num_slices, w, h) 
    :return: image, numpy array  
    '''
    sub_sample_kspace = T.to_tensor(sub_sample_kspace)      # Convert from numpy array to pytorch tensor
    sampled_image = fastmri.ifft2c(sub_sample_kspace)           # Apply Inverse Fourier Transform to get the complex image
    image = fastmri.complex_abs(sampled_image)   # Compute absolute value to get a real image
    return images #(num_slices, w, h) 

def corp_images(in_imgs, w, h):
    '''
    do center crop
    :param in_imgs: input np arrays, with dim(n, w,h)
    :param w:  new target width
    :param h:  new target height (be careful to validate the size)
    :return: out_imgs: dim(n ,new_w, new_h)
    '''
    out_imgs = None
    # https://github.com/tensorflow/datasets/pull/1041/files
    #todo:
    assert 0 < w<= in_imgs.shape[-2]
    assert 0 < h<= in_imgs.shape[-1]
    w_from = (in_imgs.shape[-2] - w) // 2
    h_from = (in_imgs.shape[-1] - h) // 2
    w_to = w_from + w
    h_to = h_from + h
    out_imgs = in_imgs[..., w_from:w_to, h_from:h_to]
    return out_imgs


def concate_images(in_imgs):
    '''
    flatten the numpy array
    input images(num_slices, w, h) -> output images(h, num_slices*w)
    :param in_imgs: numpy array
    :return:  out_imgs: numpy array
    '''
    #todo:
    #https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape
    h = in_imgs.shape[-1]
    out_imgs = in_imgs.transpose(2,0,1).reshape(h ,-1)
    return out_imgs

def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    return torch.from_numpy(data)
