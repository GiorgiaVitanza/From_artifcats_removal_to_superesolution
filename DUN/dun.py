#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2020-11-11 20:37:09
LastEditTime: 2023-07-07 10:01:54
Description: 
batch_size = 64, MSE, Adam, 0.0001, patch_size = 64, 2000 epoch, decay 1000, x0.1
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        base_filter = 64
        out_channels = args['data']['n_colors']
        num_channels = out_channels + 6
        
        self.args = args
        self.head = ConvBlock(num_channels, 48, 9, 1, 4, activation='relu', norm=None, bias = True)

        self.body = ConvBlock(48, 32, 5, 1, 2, activation='relu', norm=None, bias = True)

        self.output_conv = ConvBlock(32, out_channels, 5, 1, 2, activation='relu', norm=None, bias = True)

        self.cycle = cycle()
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                # torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, l_ms, b_ms, x_pan):

        NDWI = ((l_ms[:, 1, :, :] - l_ms[:, 3, :, :]) / (l_ms[:, 1, :, :] + l_ms[:, 3, :, :])).unsqueeze(1)
        NDWI = torch.where(torch.isnan(NDWI), torch.full_like(NDWI, 0), NDWI)
        NDWI = F.interpolate(NDWI, scale_factor=self.args['data']['upsacle'], mode='bicubic')
        NDVI = ((l_ms[:, 3, :, :] - l_ms[:, 2, :, :]) / (l_ms[:, 3, :, :] + l_ms[:, 2, :, :])).unsqueeze(1)
        NDVI = torch.where(torch.isnan(NDVI), torch.full_like(NDVI, 0), NDVI)
        NDVI = F.interpolate(NDVI, scale_factor=self.args['data']['upsacle'], mode='bicubic')
        x_pan_1, x_pan_2 = self.cycle(x_pan)
        x_f = torch.cat([b_ms, x_pan_1, NDVI, NDWI], 1)
        x_f = self.head(x_f)
        x_f = self.body(x_f)
        x_f = self.output_conv(x_f)

        return x_f, x_pan_1, x_pan_2


class cycle(nn.Module):
    def __init__(self, args):
        super(cycle, self).__init__()

        n_resblocks = 3
        
        block1 = [
            ConvBlock(1, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            block1.append(ResnetBlock(32, 5, 1, 1, 0.1, activation='prelu', norm=None))
        self.block1 = nn.Sequential(*block1)

        block2 = [
            ConvBlock(5, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            block2.append(ResnetBlock(32, 1, 1, 1, 0.1, activation='prelu', norm=None))
        self.block2 = nn.Sequential(*block2)
    
    def forward(self, x):

        y = self.block1(x)
        x1 = self.block2(y)

        return y, x1
      
if __name__ == '__main__':       
    x = torch.randn(2,4,8,8)
    y = torch.randn(2,4,32,32)
    z = torch.randn(2,1,32,32)
    arg = []
    Net = Net(arg)
    out = Net(x, y, z)
    print(out.shape)