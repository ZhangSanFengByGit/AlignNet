# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 15:22:13 2018

@author: zhangzichun
"""

import torch

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        o = self.reflection_pad(x)
        o = self.conv2d(o)

        return o