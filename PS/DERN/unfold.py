#!/usr/bin/env python
# coding=utf-8
'''
Author: Jiaming-Wang wjmecho@163.com
Date: 2023-10-25 20:31:58
LastEditTime: 2024-09-01 15:02:48
FilePath: 
Description: 
'''

import os
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        base_filter = 64
        n_layer = 8
        self.err_extract0 = HinBlock(1, base_filter, 4)
        self.h_blocks = nn.ModuleList([HBlock() for i in range(n_layer)])
        self.e_blocks = nn.ModuleList([EBlock() for i in range(n_layer)])

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

        h = b_ms
        e = self.err_extract0(x_pan)
        p = x_pan
        
        for i in range(len(self.h_blocks)):
            e = self.e_blocks[i](h, e, p)
            h = self.h_blocks[i](h, e, p)
            if i == 8:
                e1 = e
            
        return e1

class HBlock(nn.Module):
    def __init__(self):
        super(HBlock, self).__init__()

        self.w = DepthHinBlock(4, 64, 1)
        self.w_t = DepthHinBlock(1, 64, 4)
        self.lamda = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lamda.data.fill_(0.1)
        self.prox = HinBlock(4, 64, 4)

    def forward(self, h, e, p): 
        tem = self.w(h + e)
        tem = tem - p
        tem = self.lamda * self.w_t(tem)
        tem = self.prox(h - tem)
        return tem

class EBlock(nn.Module):
    def __init__(self):
        super(EBlock, self).__init__()

        self.w = DepthHinBlock(4, 64, 1)
        self.w_t = DepthHinBlock(1, 64, 4)
        self.lamda = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lamda.data.fill_(0.1)
        self.prox = HinBlock(4, 64, 4)

    def forward(self, h, e, p):
        tem = self.w(h + e) - p
        tem = self.w_t(tem)
        tem = self.lamda * tem
        tem = e - tem
        tem = self.prox(tem)
        return tem

class HinBlock(nn.Module):
    def __init__(self, in_size, mid_size, out_size, relu_slope=0.2):
        super(HinBlock, self).__init__()

        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = nn.Conv2d(in_size, mid_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(mid_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.norm = nn.InstanceNorm2d(mid_size//2, affine=True)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)
        return out

class DepthHinBlock(nn.Module):
    def __init__(self, in_size, mid_size, out_size, relu_slope=0.2):
        super(DepthHinBlock, self).__init__()

        self.identity = nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=1, stride=1, padding=0, groups=1)
        self.conv_1 = nn.Conv2d(in_channels=in_size, out_channels=mid_size, kernel_size=3, stride=1, padding=1, groups=in_size)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(in_channels=mid_size, out_channels=out_size, kernel_size=3, stride=1, padding=1, groups=out_size)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.norm = nn.InstanceNorm2d(mid_size//2, affine=True)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)
        return out

class DepthConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DepthConv, self).__init__()

        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)

        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out