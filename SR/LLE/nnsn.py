#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2021-03-03 11:02:10
LastEditTime: 2021-07-23 01:03:23
Description: batch_size=16, patch_size=48, L1 loss, epoch=250, lr=1e-5, decay=250, ADAM
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from model.nonlocal_layer_vgg import *

## Holistic Attention Network (HAN)
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        
        n_resgroups = 10
        n_resblocks = 30
        base_filter = 64
        kernel_size = 3
        reduction = 16
        num_channels = 3
        scale_factor = 4

        # define head module
        self.head = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='relu', norm=None)

        self.nn1 = NONLocalBlock2D(in_channels=base_filter, mode='embedded_gaussian')
        
        # define body module
        body = [
            ResidualGroup(base_filter, 3, reduction, act=nn.ReLU(True), res_scale=0.1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]
            
        body.append(ConvBlock(base_filter, base_filter, 3, 1, 1, activation='relu', norm=None))

        self.nn2 = NONLocalBlock2D(in_channels=base_filter, mode='embedded_gaussian')
        # define tail module
        self.up = Upsampler(scale_factor, base_filter, activation=None)
        self.output_conv = ConvBlock(base_filter, num_channels, 3, 1, 1, activation='relu', norm=None)

        # self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        # self.csa = CSAM_Module(base_filter)
        self.la = LAM_Module(base_filter)
        self.last_conv = nn.Conv2d(base_filter*11, base_filter, 3, 1, 1)
        self.last = nn.Conv2d(base_filter*2, base_filter, 3, 1, 1)


    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)
        x, att1_1, att1_2 = self.nn1(x)
        res = x
        #pdb.set_trace()
        for name, midlayer in self.body._modules.items():
            res = midlayer(res)
            if name=='0':
                res1 = res.unsqueeze(1)
            else:
                tmp = res.unsqueeze(1)
                res1 = torch.cat([tmp,res1],1)
        out1 = res
        # res3 = res.unsqueeze(1)
        # res = torch.cat([res1,res3],1)
        res, att = self.la(res1)
        out2 = self.last_conv(res)

        out1, att2_1, att2_2  = self.nn2(out1)
        out = torch.cat([out1, out2], 1)
        res = self.last(out)
        
        res += x
        # res = self.csa(res)

        x = self.up(res)
        x = self.output_conv(x)
        # x = self.add_mean(x)
        #return x, att1_1, att1_2, att2_1, att2_2, att
        return x

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size, reduction, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(ConvBlock(n_feat, n_feat, 3, 1, 1, activation=None, norm=None))
            # if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res = res + x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                n_feat, kernel_size, reduction, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(ConvBlock(n_feat, n_feat, 3, 1, 1, activation=None, norm=None))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = res + x
        return res

class LAM_Module(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim):
        super(LAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

        self.conv1 = nn.Conv3d(64, 1, 1, 1, 0)
        self.conv2 = nn.Conv3d(1, 64, 1, 1, 0)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, N, C, height, width = x.size()
        xx = x

        x_com = self.conv1(x.view(m_batchsize, C, N, height, width))
        proj_query = x_com.view(m_batchsize, N, -1)
        proj_key = x_com.view(m_batchsize, N, -1).permute(0, 2, 1)

        energy = torch.matmul(proj_key, proj_query)
        
        energy_new = torch.max(energy, -1, keepdim=True)[0]
        energy_new =energy_new.expand_as(energy)
        energy_new =energy_new -energy
        attention = self.softmax(energy_new)
        # proj_value = x_com.view(m_batchsize, C*height*width, -1)
        out = torch.bmm(attention, proj_key)
        out = out.view(m_batchsize, 1, N, height, width)
        out = self.conv2(out)
        out = out.view(m_batchsize, N, -1, height, width)
        out = self.gamma*out + xx
        out = out.view(m_batchsize, -1, height, width)
        return out, attention

class CSAM_Module(nn.Module):
    """ Channel-Spatial attention module"""
    def __init__(self, in_dim):
        super(CSAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.softmax  = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X N X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X N X N
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)
        out = self.sigmoid(self.conv(out))
        
        # proj_query = x.view(m_batchsize, N, -1)
        # proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)
        # energy = torch.bmm(proj_query, proj_key)
        # energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        # attention = self.softmax(energy_new)
        # proj_value = x.view(m_batchsize, N, -1)

        # out = torch.bmm(attention, proj_value)
        # out = out.view(m_batchsize, N, C, height, width)

        out = self.gamma*out
        out = out.view(m_batchsize, -1, height, width)
        x = x * out + x
        return x