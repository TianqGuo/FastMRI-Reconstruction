import os
import csv
import numpy as np
import Data.Data_Transform as dt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import h5py
import matplotlib.pyplot as plt
import fastmri
from fastmri.data import transforms as T
from sklearn.model_selection import train_test_split
import config_file

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
    max_limit = min(max_limit, len(files))
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


def read_h5_from_file_with_filter(path, slice_idxs=None):
    '''
    use h5py to read kspace, target, file_name
    :param path: h5 path
    :return:  H5Data
    '''
    with h5py.File(path, 'r') as h5_file:
        if slice_idxs is None:
            slice_idxs = slice(0, h5_file['kspace'].shape[0])

        k_space = np.array(h5_file['kspace'][slice_idxs])
        target = np.array(h5_file['reconstruction_rss'][slice_idxs])
        file_name = os.path.basename(path)

        print('Current h5 file keys:', list(h5_file.keys()))
        print('Current h5 file Attrs:', dict(h5_file.attrs))

    return H5Data(k_space, target, file_name)
  
def plot_data_coils(data, slice_nums, cmap=None):
    '''
    plot the kspace data
    :param k_space: kspace data
    :return:
    '''
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)
    plt.show()


def split_data(file_list, test_size=0.2, random_state=None):
    """
    Splits the list of files into training and testing sets.

    Parameters:
    - file_list: List of file paths.
    - test_size: Proportion of the dataset to include in the test split.
    - random_state: Controls the shuffling applied to the data before the split. Pass an int for reproducible output.

    Returns:
    - train_files: List of files for training.
    - test_files: List of files for testing.
    """
    train_files, test_files = train_test_split(file_list, test_size=test_size, random_state=random_state)
    return train_files, test_files

  
def create_data_loader(args, data_files, shuffle=True, ):
    '''
    Create data loaders such as train loader, validation loader and test loader for training and testing.
    Use FastMriDataset class

    Returns:
        A data loader
    '''
    # Get dataset


    # Create data loader
    data_loader = DataLoader(
        dataset= None,
        batch_size= None,
        shuffle= None,
    )

    return data_loader

def sanity_test():
    print("Current Working Directory:", os.getcwd())
    h5_file_list = get_h5_file_list(config_file.INPUT_DATA_DIR, 4)
    print(h5_file_list)

    train_files, test_files = split_data(h5_file_list, test_size=0.25, random_state=42)  # 25% data as test set
    print("Training files:", len(train_files))
    print("Testing files:", len(test_files))

    h5_data = read_h5_from_file_with_filter(h5_file_list[0], [20, 21, 22, 23])
    print(h5_data.k_space.dtype)
    print(h5_data.k_space.shape)

    # slice_kspace = h5_data.k_space[20]
    plot_data_coils(np.log(np.abs(h5_data.k_space) + 1e-9), [0, 1, 2, 3])
    slice_kspace2 = T.to_tensor(h5_data.k_space)
    slice_image = fastmri.ifft2c(slice_kspace2)
    slice_image_abs = fastmri.complex_abs(slice_image)
    plot_data_coils(slice_image_abs, [0, 1, 2, 3], cmap='gray')

'''
This class is used for creating dataloader
'''
class FastMriDataset:
    def __init__(self, h5_file_list, slice_idxs, corp_size, transform = None):
        '''
        :param h5_file_list:
        :param annotation_file_list:
        :param slices_idx: the list if slice idx we care about
        '''
        self.h5_file_list = h5_file_list
        #self.annotation_file_list = annotation_file_list
        self.slice_idxs =  slice_idxs
        self.corp_size = corp_size

    def __len__(self):
        return len(self.h5_file_list)


    def __getitem__(self, index):
        h5_file = self.h5_file_list[index]

        h5_filter = read_h5_from_file_with_filter(h5_file, self.slice_idxs)

        assert(len(self.slice_idxs) > 0)
        assert(len(h5_filter.k_space) == len(self.slice_idxs))
        slice_arr_list = []
        target_arr_list = []
        for i in range(len(h5_filter.k_space)):
            ## sub_sampled kspace dataset
            slice_kspace = h5_filter.k_space[i]
            slice_target = h5_filter.target[i]
            k_space_sub_sampled = dt.apply_sub_sample(slice_kspace)
            ## IFFT
            ifft_imgs = dt.inverse_fft(k_space_sub_sampled)
            #because the eadge area of image is dark, in order to accelerate the computing,we need to corp
            ## corp ifft images to fit the target size
            corp_w, corp_h = slice_target.shape
            corp_ifft_imgs = dt.corp_images(ifft_imgs, corp_w, corp_h)
            print(corp_ifft_imgs.shape)
            #corp_target_imgs = dt.corp_images(h5_filter.target, corp_w, corp_h)
            corp_ifft_imgs = np.abs(corp_ifft_imgs)
            corp_ifft_imgs = (corp_ifft_imgs - np.min(corp_ifft_imgs))/(np.max(corp_ifft_imgs) - np.min(corp_ifft_imgs))

            slice_target = np.abs(slice_target)
            slice_target = (slice_target- np.min(slice_target))/(np.max(slice_target) - np.min(slice_target))

            slice_arr_list.append(corp_ifft_imgs)
            target_arr_list.append(slice_target)

        ##stack arr
        slices_arr = np.stack(slice_arr_list, axis=0)
        targets_arr = np.stack(target_arr_list, axis=0)

        ##convert to tensor
        ifft_tensor = dt.to_tensor(slices_arr)
        target_tensor = dt.to_tensor(targets_arr)

        ##normalize tensor
        return(ifft_tensor, target_tensor)


if __name__ == '__main__':
    sanity_test()



