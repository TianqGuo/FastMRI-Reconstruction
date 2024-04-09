'''
THis file is used for implement common used functions
'''
import numpy
import numpy as np
import torch
import  Data.Fastmri_Data_Processing
import torch.nn as nn
import torch.nn.functional as F
import h5py



def visualize_h5_slice(h5, slice_num):
    '''
    visulize the kspace, target
    :param
    h5: an single H5Data(kspace, target, filename)
    slice_num: the specific slice we need to visualize
    :return:
    '''

def visualize_ifft_vs_target():
    #todo
    return