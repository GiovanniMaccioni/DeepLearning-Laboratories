import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from itertools import chain



class Hidden_Layer(nn.Module):
    """
    width: Number of neurons in the hidden layer
    """
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.actv =  nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.actv(x)
        return x

class MLP(nn.Module):
    """
        input_size: Dimension of the input that determines the size of the input layer
        num_hidden_layers: Number of hidden layers
        width: Number of neurons in each hidden layer
        num_classes: Number of classes to classify
        num_levels: Number of ResNet Blocks
        residual_step: How many hidden layers are skipped
    """
    def __init__(self, input_size, num_hidden_layers, width, num_classes, residual_step = 0):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Flatten(), nn.Linear(input_size, width), nn.ReLU())
        self.hidden_layers = nn.ModuleList([Hidden_Layer(width) for _ in range(num_hidden_layers)])
        self.output_layer = nn.Linear(width,num_classes)

        self.residual_step = residual_step
        self.num_hidden_layers = num_hidden_layers

        if residual_step != 0:
            assert residual_step < num_hidden_layers , "residual_step can't be above num_hidden_layers"
            assert (num_hidden_layers - 2) % residual_step == 0 , "residual_step should be a dividor for num_hidden_layers - 2"
            self.num_shortcuts = (num_hidden_layers - 2) // residual_step
            self.identities = nn.ModuleList([nn.Identity() for _ in range(self.num_shortcuts)])
             
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers[0](x)

        if self.residual_step != 0:
            count = self.num_shortcuts
            for i in range(1, len(self.hidden_layers)-2, self.residual_step):
                #print(f"i :{i}")
                ident = self.identities[-count](x)
                for j in range(i, i + self.residual_step - 1):
                    #print(f"j :{j}")
                    x = self.hidden_layers[j](x)
                x = x + ident
                count -= 1
                #print(count)
        else:
            for hl in self.hidden_layers:
                x = hl(x)

        x = self.output_layer(x)
        return x
    



    
class ResNet_Block(nn.Module):
    """
        in_channels: Number of channels of the input image
        out_channels: Number of output channels of the convolution block
        kernel_size:  Dimension of the kernel 
        stride: stride
        padding: padding     
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.actv = nn.ReLU()

    def forward(self, x, proj = None):
        x = self.conv(x)
        x = self.norm(x)
        if proj != None:
            #proj is the output of a residual connection
            x = x + proj
        x = self.actv(x)
        return x

    
class ResNet_Layer(nn.Module):
    """
        in_channels: Number of channels of the input feature map
        out_channels: Number of output channels of the convolution block
        num_conv:  Number of convolutional layers to stack 
        residual_step: Number of convolution layers to skip
        first: Indicates if it is the first ResNet Layer; it needs a different initialization
    """
    def __init__(self, in_channels, out_channels, num_conv, residual_step = 0, first = False):
        super().__init__()
        
        if first == True:
            stride_first = 1
            in_channels_first = in_channels
        else:
            stride_first = 2
            in_channels_first = in_channels // 2
        
        self.block = nn.ModuleList([ResNet_Block(in_channels_first, out_channels, 3, stride_first, padding = 1)]\
                                + [ResNet_Block(in_channels, out_channels, 3, stride = 1, padding = 1) for _ in range(num_conv-1)])
        self.residual_step = residual_step

        if residual_step != 0:
            assert residual_step <= num_conv , "residual_step can't be above num_conv"
            assert num_conv % residual_step == 0 , "residual_step should be a dividor for num_conv"
            self.num_shortcuts = num_conv // residual_step
            if first == True:
                self.projections = nn.ModuleList([nn.Identity() for _ in range(self.num_shortcuts)])
            else:
                self.projections = nn.ModuleList([nn.Conv2d(out_channels//2, out_channels, 1, 2)]\
                                                 + [nn.Identity() for _ in range(self.num_shortcuts - 1)])


    def forward(self,x):

        if self.residual_step != 0:
            count = self.num_shortcuts
            for i in range(0, len(self.block), self.residual_step):
                proj = self.projections[-count](x)
                for j in range(i, i + self.residual_step):
                    if j != (i + self.residual_step):
                        x = self.block[j](x)
                    else:
                        x = self.block[j](x, proj)
                #x = x + proj
                count -= 1
        else:
            for i in range(len(self.block)):
                x = self.block[i](x)
        
        return x
    

class ResNet(nn.Module):
    """
        in_channels: Number of channels of the input image
        out_channels: Number of output channels of the first convolution block
        num_classes: Number of classes to classify
        num_conv_per_level: Number of convolution in The ResNet Block
        num_levels: Number of ResNet Blocks
        residual_step: How many Convolutions are skipped
    """
    def __init__(self, in_channels, out_channels, num_classes, num_conv_per_level, num_levels, residual_step):
        super().__init__()
        self.conv_in = ResNet_Block(in_channels, out_channels, 7, 2, 3)
        self.pool = nn.MaxPool2d(3, 2)

        assert num_conv_per_level > 0, "0 not allowed in num_conv_per_level; at least 1"
        
        self.res_block_list = nn.Sequential(*([ResNet_Layer(out_channels, out_channels, num_conv_per_level, residual_step, first = True)]\
                                            +[ResNet_Layer(out_channels*2**i, out_channels*2**i, num_conv_per_level, residual_step) for i in range(1, num_levels)]))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(out_channels*2**(num_levels-1), num_classes)

    def forward(self,x):
        x = self.conv_in(x)
        x = self.pool(x)

        """for res_block in self.res_block_list:
            x = res_block(x)"""
        x = self.res_block_list(x)

        x = self.global_pool(x)
        x = x.reshape(x.shape[0], -1)#TOCHECK
        x = self.linear(x)

        return x
    


