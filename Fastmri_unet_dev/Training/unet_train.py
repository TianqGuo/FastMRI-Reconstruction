'''
THis file is used for implement common used functions
'''
import torch
from torch import nn
from torch.nn import functional as F

def train_epoch(args, model, train_loader, loss, optimizer):
    '''
    Training process for one single epoch
    Args:
        model: Unet model to train with
        train_loader: train loader for iterating data
        optimizer: optimizer for training process

    '''



def validate(args, model, val_loader):
    '''
    Validation process
    Args:
        model: Unet model to train with
        val_loader: validation loader for iterating data

    '''

def save_model(export_dir, ):
    '''
    Save model during training
    Args:
        export_dir: model export direction

    Returns:

    '''


def train(args, model, loss, optimizer, train_loader, val_loader, num_epochs):
    '''
    Implementing Training Process
    Args:
        model: Unet model to train with
        loss: loss metrics to train with
        optimizer: optimizer to train with
        train_loader: data loader for training
        val_loader: data loader for validation
        num_eposchs: number of training epochs

    '''
    start_epoch = 0

    for epochs in range(start_epoch, num_epochs):
        # Run single one epoch to get training loss
        train_loss,  = train_epoch(args, model, train_loader, loss, optimizer)
        # validation loss
        val_loss,  = validate(args, model, val_loader)



