import numpy

import Utils
import config_file
import  Data.Fastmri_Data_Processing as dp
from torch.utils.data import DataLoader
'''
Test the implemenation of data processing of data pre_processing 
Owner: Yawen Xiao
'''

def test_data_processing():
    '''
    The purpose of this function is given an input data dir and input annotation
    to produce a dataloader which can be pass to the modle later
    '''
    ###
    input_data_dir = config_file.INPUT_DATA_DIR
    input_annotation_dir = config_file.INPUT_ANOTATION_DIR
    slice_idxs = config_file.SLICES
    crop_size = config_file.CROP_SIZE
    batch_size = config_file.BATCH_SIZE
    max_num_data = 100

    h5_lists = dp.Get_h5_file_list(input_data_dir, max_num_data)

    annotation_lists = dp.Get_annotation_file_list(input_annotation_dir)

    fastmri_dataset = dp.FastMriDataset(h5_lists, annotation_lists, slice_idxs, crop_size)

    train_loader = DataLoader(fastmri_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    ## Read
    for ifft_img, target in train_loader:
        # it iterates each batch
        # do sth train, and update
        print(ifft_img.shape)
    return


def main():
    print("This is a test for data preprocessing")
    return

main()
