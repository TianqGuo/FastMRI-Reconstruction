'''
THis file is used for implement common used functions
'''
import torch
from torch import nn
from torch.nn import functional as F

def train_epoch(args, model, train_loader, criterion, optimizer):
    '''
    Training process for one single epoch
    Args:
        model: Unet model to train with
        train_loader: train loader for iterating data
        optimizer: optimizer for training process

    '''

    model.train()
    total_loss = 0

    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(args, model, val_loader, criterion):
    '''
    Validation process
    Args:
        model: Unet model to train with
        val_loader: validation loader for iterating data

    '''
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def save_model(model, export_dir, epoch):
    '''
    Save model during training
    Args:
        export_dir: model export direction

    Returns:

    '''
    torch.save(model.state_dict(), f'{export_dir}/model_epoch_{epoch}.pth')


def train(args, model, criterion, optimizer, train_loader, val_loader, num_epochs):
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
    export_dir = 'saved_models'

    for epoch in range(start_epoch, num_epochs):
        train_loss = train_epoch(args, model, train_loader, criterion, optimizer)
        val_loss = validate(args, model, val_loader, criterion)

        print(f'Epoch {epoch}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

    # Save the model every epoch
    save_model(model, export_dir, epoch)

    # for epochs in range(start_epoch, num_epochs):
    #     # Run single one epoch to get training loss
    #     train_loss,  = train_epoch(args, model, train_loader, loss, optimizer)
    #     # validation loss
    #     val_loss,  = validate(args, model, val_loader)






