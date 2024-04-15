"""
This code is for running the whole process
"""
import glob, os

import numpy as np
import torch

import argparse
import config_file

import Model.unet as Unet
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
    input_data_path = os.path.join(os.path.abspath('.'), 'dataset')
    h5_file_list = glob.glob(os.path.join(input_data_path, 'singlecoil_train', '*.h5'))
    export_dir = ''

    if mode == 'train':
        # Create Unet model
        model = Unet()
        model.to(device=args.device)

        # Loss type
        loss = losses.ssim_loss()
        loss.to(device=args.device)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), args.lr)

        # Train and validation data loader
        train_files, val_files = dp.split_data(file_list=h5_file_list)
        train_loader = dp.create_data_loader(args, train_files)
        val_loader = dp.create_data_loader(args, val_files)

        # Training
        unet_train.train(args, model, loss, optimizer, train_loader, val_loader)

    elif mode == 'test':

        # Load best model from saved models
        model = Unet()
        model.to(args.device)
        ...

        test_files =
        test_loader = dp.create_data_loader(args, test_files)

        # Output and save reconstructions
        reconstructions = unet_test.test(args, model, test_loader)\




if __name__ == '__main__':
    args = arg_parser()

    main(args, mode='train')




