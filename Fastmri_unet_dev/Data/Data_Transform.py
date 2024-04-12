import numpy
import numpy as np
import torch
import Fastmri_Data_Processing
import torch.nn as nn
import torch.nn.functional as F
import h5py

'''
owner: Fang Zou
'''


def apply_sub_sample(k_space: numpy.array):
    '''
    :param k_space: numpy array with dims(num_slices, w, h)
    :return: k_space_sub_sampled(num_slices, w, h ) , apply zero_ filled mask, to mask out the original k_space
    '''
    #todo
    k_space_sub_sampled = None

    return k_space_sub_sampled

def inverse_fft(sub_sample_kspace: numpy.array):
    '''
    :param sub_sample_kspace: dim(num_slices, w, h) 
    :return: image, numpy array
    '''
    inv_images = None
    return inv_images

def corp_images(in_imgs, w, h):
    '''
    do center crop
    :param in_imgs: input np arrays, with dim(n, w,h)
    :param w:  new target width
    :param h:  new target height (be careful to validate the size)
    :return: out_imgs: dim(n ,new_w, new_h)
    '''
    out_imgs = None
    #todo:
    return out_imgs

def concate_images(in_imgs):
    '''
    flatten the numpy array
    input images(num_slices, w, h) -> output images(h, num_slices*w)
    :param in_imgs: numpy array
    :return:  out_imgs: numpy array
    '''
    #todo:
    out_imgs = None
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

