"""
This code is for running the whole process
"""
import glob, os

import numpy as np
import torch

import argparse

from torch.utils.data import DataLoader

import config_file

# import Model.unet as Unet
from fastmri.models import Unet
import Data.Fastmri_Data_Processing as dp
import Training.losses as losses
import Training.unet_train as unet_train
import Testing.unet_test as unet_test

import Utils

def arg_parser():
    parser = argparse.ArgumentParser(description='Train Unet for MRI Reconstruction'
                                     , formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', default=config_file.BATCH_SIZE, type=int, help='Batch Size')
    parser.add_argument('--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('--num-epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--in-chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out-chans', type=int, default=1, help='Size of output channels for network')
    parser.add_argument('--device', type=str, default='cuda', help='device to run on')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    args = parser.parse_args()
    return args

def main(args, mode='train'):
    # input_data_path = os.path.join(os.path.abspath('.'), 'dataset')
    # h5_file_list = glob.glob(os.path.join(input_data_path, 'singlecoil_train', '*.h5'))
    # export_dir = ''

    input_data_dir = config_file.INPUT_DATA_DIR
    input_validation_dir = config_file.INPUT_VALID_DATA_DIR
    input_annotation_dir = config_file.INPUT_ANOTATION_DIR
    slice_idxs = config_file.SLICES
    crop_size = config_file.CROP_SIZE
    batch_size = config_file.BATCH_SIZE
    max_num_data = config_file.MAX_FILE_LIMIT

    h5_lists = dp.get_h5_file_list(input_data_dir, max_num_data)
    h5_validate_lists = dp.get_h5_file_list(input_validation_dir, max_num_data)

    if input_annotation_dir != '':
        annotation_lists = dp.get_annotation_file_list(input_annotation_dir)

    if len(h5_lists) == 0:
        print("There is no h5 file input.")
        return

    fastmri_dataset = dp.FastMriDataset(h5_lists, slice_idxs, crop_size)

    fastmri_validate_dataset = dp.FastMriDataset(h5_validate_lists, slice_idxs, crop_size)

    train_loader = DataLoader(fastmri_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    validation_loader = DataLoader(fastmri_validate_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    num_channel = train_loader.dataset[0][0].shape[0]

    print("num_channel: ", num_channel)

    model = Unet(in_chans=num_channel,
                 out_chans=num_channel,
                 chans=32,
                 num_pool_layers=4,
                 drop_prob=0.0)
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    if mode == 'train':
        # Training
        unet_train.train(args, model, True, optimizer, train_loader, validation_loader)

    elif mode == 'test':

        test_files = None
        test_loader = dp.create_data_loader(args, test_files)

        # Output and save reconstructions
        reconstructions = unet_test.test(args, model, test_loader)\




if __name__ == '__main__':
    args = arg_parser()

    main(args, mode='train')




