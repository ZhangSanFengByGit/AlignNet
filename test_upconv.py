# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 15:00:14 2018

@author: zhangzichun
"""

import torch as t
from UpsampleConvLayer import UpsampleConvLayer
from PIL import Image
from torchvision import transforms

to_tensor=transforms.ToTensor()
to_img=transforms.ToPILImage()

reflect=t.nn.ReflectionPad2d(50)

img=Image.open('test.jpg')
img=img.resize([224,224])
img=to_tensor(img)
img=img.unsqueeze(0)

img=reflect(img)
#up=UpsampleConvLayer(3,3,3,1,2)
#img=up(img)

img=to_img(img.data[0])