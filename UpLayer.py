

import torch
from UpsampleConvLayer import UpsampleConvLayer
from ConvLayer import ConvLayer
from ResidualBlock import ResidualBlock

class UpLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsample=2):
        super(UpLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        #reflection_padding = kernel_size // 2
        #self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        #self.conv1 = ConvLayer(in_channels, in_channels, kernel_size, stride)
        self.conv1 = ResidualBlock(in_channels)
        self.conv2 = ConvLayer(in_channels, out_channels, kernel_size ,stride)
        #self.conv3 = ConvLayer(out_channels,out_channels,kernel_size,stride)
        self.conv3 = ResidualBlock(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        #residual = x
        o = x
        if self.upsample:
            o = self.upsample_layer(o)
        #o = self.reflection_pad(o)
        o = self.conv1(o)
        #o = self.relu(o)
        o = self.conv2(o)
        o = self.relu(o)
        o = self.conv3(o)
        #o = o + residual

        return o
    
    
    
    
    
    
    