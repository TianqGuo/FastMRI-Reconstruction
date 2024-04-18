'''
This file is Pytorch implementation of Unet Model
'''
import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    '''
    A convolution block implemented for conveniently constructing both Down-sampling and Up-sampling of Unet Model

    Each convolution block contains two convolution layers each followed by one normalization layer, one ReLU activation layer,
    and a dropout layer. (Conv1, norm, ReLU, dropout, Conv2, norm, ReLU, dropout)
    '''

    def __int__(self, in_chans, out_chans, drop_prob):
        '''
        Args:
            in_chans: number of input channels for Convolution Block
            out_chans: number of output channels for Convolution Block
            drop_prob: Dropout probability
        '''

    def forward(self, input):
        '''
        Args:
            input: torch tensor of shape (N, in_chans, H, W), N is batch size

        Returns:
            output: torch tensor of shape (N, out_chans, H, W)
        '''

class TransposeConvBlock(nn.Module):
    '''
    A Transpose Convolution Block implemented for conveniently constructing Up-sampling of Unet Model

    Each transpose convolution block contains a convolution layer followed by a normalization layer and a ReLU activation layer.
    '''
    def __int__(self, in_chans, out_chans):
        '''
        Args:
            in_chans: number of input channels for Transpose ConvBlock
            out_chans: number of output channels for Transpose ConvBlock
        '''


    def forward(self, input):
        '''
        Args:
            input: input tensor of shape (N, in_chans, H, W)

        Returns: output: output tensor of shape (N, out_chans, H*2, W*2)

        '''



class Unet(nn.Module):

    def __int__(self, in_chans, out_chans, chans, num_layers, drop_prob):
        '''
        Unet Model
        Args:
            in_chans: number of input channels
            out_chans: number of output channels
            chans: number of output channel for the first convolution layer
            num_layers: number of down-sampling or up-sampling layers
            drop_prob: dropout probability
        '''



    def forward(self, input):
        '''
        Args:
            input (torch tensor): input image of shape (N, C, H, W), N is batch size

        Returns:
            output (torch tensor): output tensor of shape (N, C, H, W), N is batch size
        '''






