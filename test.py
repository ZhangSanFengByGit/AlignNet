# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 22:52:08 2018

@author: cvpr
"""
from AlignNet import AlignNet

import torch
from torchvision import transforms
from PIL import Image

to_img = transforms.ToPILImage()

transform = transforms.Compose([
        transforms.Resize(550),
        transforms.RandomResizedCrop(512),
        transforms.RandomRotation([-15,15]),
        transforms.RandomHorizontalFlip(0.2),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

transform2 = transforms.Compose([
        transforms.Resize([512,512]),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

img1 = Image.open('H:\\pedestrian_RGBT\\kaist-rgbt\\images\\set1\\V000\\visible\\I00000.jpg')
img2 = Image.open('H:\\pedestrian_RGBT\\kaist-rgbt\\images\\set1\\V000\\lwir\\I00000.jpg')

img1 = transform2(img1).unsqueeze(0).cuda()
img2_trans = transform(img2).unsqueeze(0).cuda()


alignTest = AlignNet()
alignTest = alignTest.cuda()

#temp Area~~~~~~~~
alignTest.load_state_dict(torch.load('H:\\model_checkPoint_save_version2\\ckpt_epoch_1_Sequence_id_35.plk'))

out = alignTest(img1,img2_trans)

out = out.cpu()
out = out.data[0]
#out = to_img(out)










