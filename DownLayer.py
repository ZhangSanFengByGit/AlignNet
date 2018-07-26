
from ResidualBlock import ResidualBlock
from ConvLayer import ConvLayer
import torch

class DownLayer(torch.nn.Module):
    def __init__(self,inchannel,outchannel,kernel_size=3,stride=1):
        super(DownLayer,self).__init__()
        #self.conv1=ConvLayer(inchannel,inchannel,kernel_size,stride=1 )
        self.conv1 = ResidualBlock(inchannel)
        #self.bn1 = torch.nn.BatchNorm2d(inchannel, affine=True )
        self.conv2 = ConvLayer(inchannel, outchannel, kernel_size, stride =2 )
        self.bn2 = torch.nn.BatchNorm2d(outchannel, affine=True )
        #self.conv3=ConvLayer(outchannel,outchannel,kernel_size,stride =1)
        self.conv3 = ResidualBlock(outchannel)
        #self.bn3 = torch.nn.BatchNorm2d(outchannel , affine=True )
        self.relu = torch.nn.ReLU(inplace=True)


    def forward(self,x):
    	o = self.conv1(x)
    	#o = self.bn1(o)
    	#o = self.relu(o)
        
    	o = self.conv2(o)
    	o = self.bn2(o)
    	o = self.relu(o)
        
    	o = self.conv3(o)
    	#o = self.bn3(o)
    	return o
