import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

from fastmri.data.subsample import RandomMaskFunc
import fastmri
from fastmri.data import transforms as T
from matplotlib import pyplot as plt

# import cv2
'''
owner: Fang Zou
'''


def apply_sub_sample(k_space: numpy.array):
    '''
    :param k_space: numpy array with dims( w, h)
    :return: k_space_sub_sampled( w, h ) , apply zero_ filled mask, to mask out the original k_space
    '''
    '''
    mask_func = RandomMaskFunc(center_fractions=[0.04], accelerations=[8])
    k_space_sub_sampled, mask, _ = T.apply_mask(k_space, mask_func) 
    return k_space_sub_sampled
    '''
    sampling_mask = np.zeros_like(k_space, dtype= bool)
    # determine the row and range of central area
    rows = k_space.shape[0]
    cols = k_space.shape[1]

    central_rows = rows//8
    central_cols = cols//8

    rows_start = (rows - central_rows)//2
    rows_end = rows_start + central_rows

    cols_start = (cols - central_cols)//2
    cols_end = cols_start +central_cols
    print(rows_start, rows_end)
    print(cols_start,cols_end)
    sampling_mask[::2, ::2] = False
    sampling_mask[rows_start:rows_end, cols_start:cols_end] = True

    subsampled_k_space = k_space.copy()
    subsampled_k_space[~sampling_mask] = 0

    return subsampled_k_space



def inverse_fft(sub_sample_kspace: numpy.array):
    '''
    :param sub_sample_kspace: dim(num_slices, w, h)
    :return: image, numpy array
    '''
    #sub_sample_kspace = T.to_tensor(sub_sample_kspace)      # Convert from numpy array to pytorch tensor
    #sampled_image = fastmri.ifft2c(sub_sample_kspace)           # Apply Inverse Fourier Transform to get the complex image
    #image = fastmri.complex_abs(sampled_image)   # Compute absolute value to get a real image
    #return image #(num_slices, w, h)
    recons_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(sub_sample_kspace, axes=(-2,-1)), norm='ortho'),axes=(-2,-1))
    return recons_img

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


def test_sub_sampled():
    path_to_h5 = '/home/cy/Documents/Deep_learning/proj/FastMRI-Reconstruction/Fastmri_unet_dev/Data/test_h5_folder/file1000000.h5'
    fig = plt.figure
    h5 = h5py.File(path_to_h5, 'r')
    k_space = np.array(h5 ['kspace'][20])

    k_space_mag = np.log(np.abs(k_space))
    plt.imshow(k_space_mag, cmap = 'gray')
    plt.show()
    #k_space_mag_norm = cv2.normalize(k_space_mag, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)

    print('The Original shape of k space:', k_space.shape)

    subsample_k_space = apply_sub_sample(k_space)

    sub_k_space_mag = np.abs(subsample_k_space)
    #sub_k_space_mag_norm = cv2.normalize(sub_k_space_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    print('The sub sampled k space: ', subsample_k_space.shape)
    plt.imshow(sub_k_space_mag, cmap='gray')
    plt.show()

def test_ifft():
    path_to_h5 = '/home/cy/Documents/Deep_learning/proj/FastMRI-Reconstruction/Fastmri_unet_dev/Data/test_h5_folder/file1000000.h5'
    fig = plt.figure
    h5 = h5py.File(path_to_h5, 'r')
    slice_idxs = 2
    k_space = np.array(h5['kspace'][slice_idxs])
    print(k_space.shape)
    target = np.array(h5['reconstruction_rss'][slice_idxs])
    recons = inverse_fft(k_space)
    recons = corp_images(recons, target.shape[1], target.shape[0])
    print(recons.shape)
    plt.imshow(target, cmap ='gray')
    print(target.shape)
    plt.show()
    plt.imshow(np.abs(recons), cmap='gray')
    plt.show()

def test_sub_ifft():
    path_to_h5 = '/home/cy/Documents/Deep_learning/proj/FastMRI-Reconstruction/Fastmri_unet_dev/Data/test_h5_folder/file1000000.h5'
    h5 = h5py.File(path_to_h5, 'r')
    k_space = np.array(h5['kspace'][20])
    orig_recons = inverse_fft(k_space)
    plt.subplot(1,2,1)
    plt.imshow(np.abs(orig_recons), cmap='gray')
    # plt.show()
    # k_space_mag_norm = cv2.normalize(k_space_mag, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)

    print('The Original shape of k space:', k_space.shape)

    subsample_k_space = apply_sub_sample(k_space)
    sub_recons = inverse_fft(subsample_k_space)
    plt.subplot(1,2,2)
    plt.imshow(np.abs(sub_recons), cmap='gray')
    plt.show()
    #sub_k_space_mag = np.abs(subsample_k_space)

def getAllAttrs():
    path_to_h5 = '/home/cy/Documents/Deep_learning/proj/FastMRI-Reconstruction/Fastmri_unet_dev/Data/test_h5_folder/file1000000.h5'
    h5 = h5py.File(path_to_h5, 'r')
    for key in h5.keys():
        print(key)

if __name__ == '__main__':
    test_ifft()
