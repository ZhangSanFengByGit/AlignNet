# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch

class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        o = x
        if self.upsample:
            o = self.upsample_layer(o)
        o = self.reflection_pad(o)
        o = self.conv2d(o)

        return o
    
    