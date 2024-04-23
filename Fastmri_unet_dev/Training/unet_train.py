'''
THis file is used for implement common used functions
'''
import os

import torch
from fastmri.models import Unet
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from Training.losses import ssim_loss
from Training.losses import mse_loss

import config_file
import pickle


def train_epoch(args, model, train_loader, loss_ssim, optimizer):
    '''
    Training process for one single epoch
    Args:
        model: Unet model to train with
        train_loader: train loader for iterating data
        optimizer: optimizer for training process

    '''

    model.train()
    total_loss = 0

    for i, batch in enumerate(train_loader):
        inputs, targets = batch[0], batch[1]
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        print("Current Batch number: ", i)
        print("inputs.shape: ", inputs.shape)
        print("targets.shape: ", inputs.shape)
        # inputs = complex_to_channels(inputs)
        # print("inputs.shape: ", inputs.shape)
        # print("targets.shape: ", inputs.shape)
        inputs = complex_to_magnitude(inputs)
        targets = complex_to_magnitude(targets)
        inputs = inputs.float()
        targets = targets.float()

        optimizer.zero_grad()
        outputs = model(inputs)
        if loss_ssim:
            loss = ssim_loss(outputs, targets)
        else:
            loss = mse_loss(outputs, targets)
        loss.to(device=args.device)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def complex_to_magnitude(data):
    # Ensure the input is complex and has an imaginary part
    if torch.is_complex(data):
        magnitude = torch.abs(data)
    else:
        magnitude = data  # Assuming data is already real
    return magnitude


def validate(args, model, val_loader, loss_ssim, threshold_ssim=0.9, threshold_mse=0.1):
    '''
    Validation process
    Args:
        model: Unet model to train with
        val_loader: validation loader for iterating data

    '''
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():  # Disable gradient computation for efficiency and to prevent model from learning
        for batch in val_loader:
            inputs, targets = batch[0], batch[1]
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            # Convert inputs and targets to their magnitudes if they're complex
            inputs = complex_to_magnitude(inputs)
            targets = complex_to_magnitude(targets)
            inputs = inputs.float()
            targets = targets.float()

            outputs = model(inputs)
            if loss_ssim:
                loss = ssim_loss(outputs, targets)
            else:
                loss = mse_loss(outputs, targets)

            total_loss += loss.item()

            # Compute "accuracy"
            diff = torch.abs(outputs - targets)
            correct_pixels += (diff < threshold_mse).sum().item()
            total_pixels += torch.numel(diff)

        average_loss = total_loss / len(val_loader)
        accuracy = correct_pixels / total_pixels

    print(f'Validation Loss: {average_loss}, MSE Accuracy: {accuracy}')

    return total_loss / len(val_loader)


def save_model(model, export_dir):
    '''
    Save model during training
    Args:
        export_dir: model export direction

    Returns:

    '''
    os.makedirs(export_dir, exist_ok=True)

    file_path = f'{export_dir}/model_epoch.pkl'

    # Open a file to write the pickled data
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)

    print(f"Model saved to {file_path}")

    return file_path


def train(args, model, loss_ssim, optimizer, train_loader, val_loader):
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
    export_dir = config_file.OUTPUT_DIR
    print("export_dir: ", export_dir)

    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, config_file.EPOCHS):
        train_loss = train_epoch(args, model, train_loader, loss_ssim, optimizer)
        val_loss = validate(args, model, val_loader, loss_ssim, config_file.THRESHOLD_SSIM, config_file.THRESHOLD_MSE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}')
        save_model(model, export_dir)

    # Plotting after training is complete
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()



