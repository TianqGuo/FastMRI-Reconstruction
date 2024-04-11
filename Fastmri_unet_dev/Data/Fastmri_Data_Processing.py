import os
import csv
import numpy as np
import Data_Transform as dt
import torch.nn as nn
import torch.nn.functional as F
import h5py

'''
Data processing for FastMRI data
owner: Tianquan Guo
'''

class H5Data:
    def __init__(self, k_space, target, file_name):
        self.k_space = k_space
        self.target = target
        self.file_name = file_name


def get_h5_file_list(in_dir, max_limit):
    '''
    input:
    indir: the dir containing h5 files
    max_limit: how many files need to process
    return : list (full path of h5 file)
    '''
    files = []
    for file in os.listdir(in_dir):
        if file.endswith('.h5'):
            files.append(os.path.join(in_dir, file))

    return files[:max_limit]


def get_annotation_file_list(in_dir):
    '''
    :param in_dir: the dir containing annotation file
    :return:
    '''
    files = []
    for file in os.listdir(in_dir):
        files.append(os.path.join(in_dir, file))

    return files


def get_annotation_data_from_csv(files):
    """
    Read annotations from a list of CSV files.

    Parameters:
    - files: List of paths to annotation CSV files.

    Returns:
    - List of dicts, where each dict contains 'file', 'slice', 'x', 'y', 'width', 'height', and 'label' for an annotation.
    """
    annotation_data = []
    for file in files:
        with open(file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                roi_data = {
                    'file': row['file'],
                    'slice': int(row['slice']),
                    'x': int(row['x']),
                    'y': int(row['y']),
                    'width': int(row['width']),
                    'height': int(row['height']),
                    'label': row['label']
                }
                annotation_data.append(roi_data)
    return annotation_data


def create_annotation_binary_mask(image_height, image_width, x, y, width, height):
    """
    Create a binary mask for a given region of interest.

    Parameters:
    - image_height: The height of the input image.
    - image_width: The width of the input image.
    - x, y: The top-left coordinates of the region of interest.
    - width, height: The dimensions of the region of interest.

    Returns:
    - A binary mask as a 2D numpy array with the same dimensions as the input image.
    """
    mask = np.zeros((image_height, image_width))

    # ROI binary mask
    mask[y:y + height, x:x + width] = 1

    return mask


def read_h5_from_file_with_filter(path, slice_idxs):
    '''
    use h5py to read kspace, target, file_name
    :param path: h5 path
    :return:  H5Data
    '''
    with h5py.File(path, 'r') as h5_file:
        k_space = np.array(h5_file['kspace'])
        target = np.array(h5_file['target'])
        file_name = os.path.basename(path)
    return H5Data(k_space[slice_idxs], target[slice_idxs], file_name)

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

        h5_filter = read_h5_from_file_with_filter(h5_file, self.slice_idxs)

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





