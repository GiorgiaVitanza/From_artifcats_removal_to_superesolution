#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2020-11-25 00:34:50
LastEditTime: 2021-12-23 10:18:36
Description: file content
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch.nn.functional as F
from model.nonlocal_patch import *
import numpy as np
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, args):
        super(Net, self).__init__()

        out_channels = 4
        n_resblocks = 9

        res_block_s1 = [
            ConvBlock(num_channels, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            res_block_s1.append(ResnetBlock_group(32, 3, 1, 1, 0.1, activation='prelu', norm=None, groups=32))
        res_block_s1.append(Upsampler(2, 32, activation='prelu'))
        res_block_s1.append(ConvBlock(32, 4, 3, 1, 1, activation='prelu', norm=None, bias = False))
        self.res_block_s1 = nn.Sequential(*res_block_s1)

        res_block_s2 = [
            ConvBlock(num_channels+1, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            res_block_s2.append(ResnetBlock_group(32, 3, 1, 1, 0.1, activation='prelu', norm=None, groups=32))
        res_block_s2.append(ConvBlock(32, 4, 3, 1, 1, activation='prelu', norm=None, bias = False))
        self.res_block_s2 = nn.Sequential(*res_block_s2)
        
        res_block_s3 = [
            ConvBlock(num_channels, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            res_block_s3.append(ResnetBlock_group(32, 3, 1, 1, 0.1, activation='prelu', norm=None, groups=32))
        res_block_s3.append(Upsampler(2, 32, activation='prelu'))
        res_block_s3.append(ConvBlock(32, 4, 3, 1, 1, activation='prelu', norm=None, bias = False))
        self.res_block_s3 = nn.Sequential(*res_block_s3)

        res_block_s4 = [
            ConvBlock(num_channels+1, 32, 3, 1, 1, activation='prelu', norm=None, bias = False),
        ]
        for i in range(n_resblocks):
            res_block_s4.append(ResnetBlock_group(32, 3, 1, 1, 0.1, activation='prelu', norm=None, groups=32))
        res_block_s4.append(ConvBlock(32, 4, 3, 1, 1, activation='prelu', norm=None, bias = False))
        self.res_block_s4 = nn.Sequential(*res_block_s4)

        ## Embedding
        self.em = NONLocalBlock2D(in_channels=1)
        self.em1 = NONLocalBlock2D(in_channels=1)
        self.em2 = NONLocalBlock2D(in_channels=1)
        self.em3 = NONLocalBlock2D(in_channels=1)

        ## Embedding
        self.em_2 = NONLocalBlock2D(in_channels=1)
        self.em1_2 = NONLocalBlock2D(in_channels=1)
        self.em2_2 = NONLocalBlock2D(in_channels=1)
        self.em3_2 = NONLocalBlock2D(in_channels=1)

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

    def functional_conv2d(self, im):
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') #
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        weight = Variable(torch.from_numpy(sobel_kernel))
        edge_detect = F.conv2d(Variable(im.cpu()), weight, padding=1)
        #edge_detect = edge_detect.squeeze().detach().numpy()
        return edge_detect.cuda()

    def img_gradient_total(self, img):
        a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        a = torch.from_numpy(a).float().unsqueeze(0)
        a = a.repeat(1, 1, 1, 1)
        # a = torch.stack((a, a, a, a))
        conv1.weight = nn.Parameter(a, requires_grad=False)
        conv1 = conv1.cuda()
        G_x = conv1(img)

        b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        b = torch.from_numpy(b).float().unsqueeze(0)
        b = b.repeat(1, 1, 1, 1)
        conv2.weight = nn.Parameter(b, requires_grad=False)
        conv2 = conv2.cuda()
        G_y = conv2(img)

        G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
        return G

    def forward(self, l_ms, b_ms, x_pan):

        hp_pan_4 =  self.img_gradient_total(x_pan)
        lr_pan = F.interpolate(x_pan, scale_factor=1/2, mode='bicubic')
        hp_pan_2 =  self.img_gradient_total(lr_pan)
        
        s1 = self.res_block_s1(l_ms)
        s1 = s1 + F.interpolate(l_ms, scale_factor=2, mode='bicubic')

        r_l_ms = torch.unsqueeze(s1[:,0,:,:],1)
        g_l_ms = torch.unsqueeze(s1[:,1,:,:],1)
        b_l_ms = torch.unsqueeze(s1[:,2,:,:],1)
        n_l_ms = torch.unsqueeze(s1[:,3,:,:],1)

        # print(r_l_ms.shape)
        r_ms_pan_2, pan_r = self.em(r_l_ms, hp_pan_2, lr_pan)
        g_ms_pan_2, pan_g = self.em1(g_l_ms, hp_pan_2, lr_pan)
        b_ms_pan_2, pan_b = self.em2(b_l_ms, hp_pan_2, lr_pan)
        n_ms_pan_2, pan_n = self.em3(n_l_ms, hp_pan_2, lr_pan)
        #a1 = -torch.log(torch.sum(a1,-1)/int(a1.size(2)))
        #a2 = -torch.log(torch.sum(a2,-1)/int(a2.size(2)))
        #s3=-torch.log(torch.sum(a3,-1)/int(a3.size(2)))
        #a4=-torch.log(torch.sum(a4,-1)/int(a4.size(2)))

        ms_pan = torch.cat([r_ms_pan_2, g_ms_pan_2, b_ms_pan_2, n_ms_pan_2], 1)
        pan2ms = torch.cat([pan_r, pan_g, pan_b, pan_n], 1)
        
        s2 = self.res_block_s2(torch.cat([s1, lr_pan], 1)) + \
            F.interpolate(l_ms, scale_factor=2, mode='bicubic') + ms_pan
        s3 = self.res_block_s3(s2) + b_ms
        
        r_l_ms = torch.unsqueeze(s3[:,0,:,:],1)
        g_l_ms = torch.unsqueeze(s3[:,1,:,:],1)
        b_l_ms = torch.unsqueeze(s3[:,2,:,:],1)
        n_l_ms = torch.unsqueeze(s3[:,3,:,:],1)
        
        r_ms_pan_2, pan_r = self.em_2(r_l_ms, hp_pan_4, x_pan)
        g_ms_pan_2, pan_g = self.em1_2(g_l_ms, hp_pan_4, x_pan)
        b_ms_pan_2, pan_b = self.em2_2(b_l_ms, hp_pan_4, x_pan)
        n_ms_pan_2, pan_n = self.em3_2(n_l_ms, hp_pan_4, x_pan)

        #a5 = -torch.log(torch.sum(a5,-1)/int(a5.size(2)))
        #a6=-torch.log(torch.sum(a6,-1)/int(a6.size(2)))
        #a7=-torch.log(torch.sum(a7,-1)/int(a7.size(2)))
        #a8=-torch.log(torch.sum(a8,-1)/int(a8.size(2)))
        ms_pan = torch.cat([r_ms_pan_2, g_ms_pan_2, b_ms_pan_2, n_ms_pan_2], 1)
        # pan2ms = torch.cat([pan_r, pan_g, pan_b, pan_n], 1)
        
        s4 = self.res_block_s4(torch.cat([s3, x_pan], 1))+ \
            b_ms + ms_pan
        return s4



if __name__ == '__main__':       
    x = torch.randn(1,4,64,64)
    y = torch.randn(1,4,256,256)
    z = torch.randn(1,1,256,256)
    arg = []
    import time
    t0= time.time
    Net = Net(arg)
    out = Net(x, y, z)
    t1 = time.time
    print(t1-t0)

