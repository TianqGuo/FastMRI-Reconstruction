'''
THis file is used for implement common used functions
'''
import numpy
import numpy as np
import matplotlib as plt

import torch
import Data.Fastmri_Data_Processing
import Data.Data_Transform as dt
import torch.nn as nn
import torch.nn.functional as F
import h5py


def visualize_h5_slice(h5, slice_nums):
    '''
    visualize the k-space, target
    :param
    h5: a single H5Data(k-space, target, filename)
    slice_num: the specific slices we need to visualize
    :return:
    '''

    ### Get volumn k-space
    filename = h5.file_name
    hf = h5py.File(filename)
    volume_kspace = hf['kspace'][()]                # shape (number of slices, height, width)

    ### visualize absolute value of k-space
    data = np.log(np.abs(volume_kspace) + 1e-9)
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i+1)
        plt.imshow(data[num], cmap=None)gi

def visualize_ifft_vs_target(ifft_img, target_img, slice_num):
    '''
    Visualize ifft image vs target image

    Args:
        ifft_img (numpy ndarray or torch tensor): image after applying Inverse Fourier Transform
        target_img (numpy ndarray): hdf5 dataset of volume (num_slices, width, height)
    '''

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(ifft_img[slice_num], cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(target_img[slice_num], cmap='gray')



