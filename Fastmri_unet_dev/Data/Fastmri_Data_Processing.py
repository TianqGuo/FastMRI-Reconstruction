import numpy as np
import Data_Transform as dt
import torch.nn as nn
import torch.nn.functional as F
import h5py

class H5Data:
    k_space: np.array
    target: np.array
    file_name: str

def Get_h5_file_list(in_dir, max_limit):
    '''
    input:
    indir: the dir containing h5 files
    max_limit: how many files need to process
    return : list (full path of h5 file)
    '''
    #Todo:
    return list()

def Get_annotation_file_list(in_dir):
    '''
    :param in_dir: the dir containing annotation file
    :return:
    '''
    #Todo:
    return list

def Read_H5_from_file(path):
    '''
    use h5py to read kspace, target, file_name
    :param path: h5 path
    :return:  H5Data
    '''
    #Todo:
    h5data = None
    return h5data

def Filter_slices(h5, silice_idxs):
    '''
    edit h5 object, only keep the slices we care about
        idxs: [0,1]
        original kspace(10,w,h) -> filtered kspace(2,w,h)
    :param h5:
    :param silice_idxs:
    :return: h5
    '''
    h5_filtered = h5
    #Todo:
    return h5_filtered

'''
This class is used for creating dataloader
'''
class FastMriDataset:
    def __init__(self, h5_file_list, annotation_file_list, slice_idxs, corp_size, transform = None):
        '''
        :param h5_file_list:
        :param annotation_file_list:
        :param slices_idx: the list if slice idx we care about
        '''
        self.h5_file_list = h5_file_list
        self.annotation_file_list = annotation_file_list
        self.slice_idxs =  slice_idxs
        self.corp_size = corp_size

    def __len__(self):
        return len(self.h5_file_list)


    def __getitem__(self, index):
        h5_file = self.h5_file_list[index]
        h5 = Read_H5_from_file(h5_file)

        ## filter slices we are interested about
        h5_filter = Filter_slices(h5, self.slice_idxs)

        ## sub_sampled kspace dataset

        k_space_sub_sampled = dt.apply_sub_sample(h5_filter.k_space)

        ## IFFT
        ifft_imgs = dt.inverse_fft(k_space_sub_sampled)

        ##because the eadge area of image is dark, in order to accelerate the computing,we need to corp
        ## inversed images and target images

        ## corp both target  and ifft images
        corp_w, corp_h = self.corp_size
        corp_ifft_imgs = dt.corp_images(ifft_imgs, corp_w, corp_h)
        corp_target_imgs = dt.corp_images(h5_filter.target, corp_w, corp_h)

        ## concatenate couple of ifft imgs to one large np array, same for targets
        ifft_slice_combined = dt.concate_images(corp_ifft_imgs)
        target_slice_combined = dt.concate_images(corp_target_imgs)

        ##convert to tensor
        ifft_tensor = dt.to_tensor(ifft_slice_combined)
        target_tensor = dt.to_tensor(target_slice_combined)

        ##normalize tensor
        return(ifft_tensor, target_tensor)





