#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2020-10-16 17:02:24
LastEditTime: 2021-08-11 00:25:13
Description: file content
'''
import torch
from torch import nn
from torch.nn import functional as F
from model.base_net import *

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.k = 5
        self.dimension = dimension
        self.sub_sample = sub_sample
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        ## Domain migration
        self.encoder = Encoder()
        # self.decoder = Decoder()

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.g1 = conv_nd(in_channels=self.in_channels, out_channels=1,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=3, stride=2, padding=1)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=3, stride=2, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            # self.g1 = nn.Sequential(self.g1, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x_ms, hp_pan, x_pan):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x_ms.size(0)
        #x_ms = self.encoder(x_ms)
        #x_pan = self.encoder(x_pan)
        #hp_pan = self.encoder(hp_pan)	
        # x_pan = self.decoder(self.encoder(x_pan))
        g_x = self.g(hp_pan).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.up(self.theta(x_ms)).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x_pan)
        phi_x = self.up(phi_x)
        phi_x = phi_x.view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
        
        zero_tensor = torch.zeros_like(f).to(f.device)
        score_f_div_C, idx = torch.topk(f, self.k, dim=2, largest=True, sorted=True)
        score_f_div_C = F.softmax(score_f_div_C, dim=-1)
        score_f_div_C = zero_tensor.scatter_(dim=2, index=idx, src=score_f_div_C)

        #zero_tensor1 = torch.zeros_like(f).to(f.device)
        

        y = torch.matmul(score_f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x_ms.size()[2:])
        W_y = self.W(y)
        z = W_y

        #score_f_div_C = F.softmax(f, dim=-1)
        #score_f_div_C = zero_tensor.scatter_(dim=2, index=idx, src=score_f_div_C)

        return z, x_pan

class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        nf = 16

        self.conv1 = ConvBlock(1, nf, 3, 1, 1, activation='prelu', norm=None, bias = False, groups=1)
        #self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = ConvBlock(nf, nf, 3, 1, 1, activation='prelu', norm=None, bias = False, groups=nf)
        #self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = ConvBlock(nf, nf, 3, 1, 1, activation='prelu', norm=None, bias = False, groups=nf)
        #self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv4 = ConvBlock(nf, nf, 3, 1, 1, activation='prelu', norm=None, bias = False, groups=nf)
    
    def forward(self, x):
        x = self.conv1(x)
        #x = self.pool1(x)
        x = self.conv2(x)
        #x = self.pool2(x)
        x = self.conv3(x)
        #x = self.pool3(x)
        x = self.conv4(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        nf = 32

        self.conv1 = ConvBlock(nf*8, nf*4, 3, 1, 1, activation='prelu', norm=None, bias = False, groups=nf*4)
        self.up1 = nn.Upsample(scale_factor=2,mode='bilinear')
        
        self.conv2 = ConvBlock(nf*4, nf*4, 3, 1, 1, activation='prelu', norm=None, bias = False, groups=nf*4)
        self.up2 = nn.Upsample(scale_factor=2,mode='bilinear')

        self.conv3 = ConvBlock(nf*4, nf*2, 3, 1, 1, activation='prelu', norm=None, bias = False, groups=nf*2)
        self.up3 = nn.Upsample(scale_factor=2,mode='bilinear')

        self.conv4 = ConvBlock(nf*2, 1, 3, 1, 1, activation='prelu', norm=None, bias = False, groups=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.up1(x)
        x = self.conv2(x)
        x = self.up2(x)
        x = self.conv3(x)
        x = self.up3(x)
        x = self.conv4(x)
        return x
