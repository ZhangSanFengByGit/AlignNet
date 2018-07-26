# Data augmentation and normalization for training
# Just normalization for validation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os

data_transforms = transforms.Compose([
        transforms.Resize(512),
        transforms.RandomResizedCrop(500),
        transforms.RandomRotation([-15,15]),
        transforms.RandomHorizontalFlip(0.2),
        transforms.ToTensor()
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = 'H:\pedestrian_RGBT\kaist-rgbt\images\set00\V000'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['lwir', 'visible']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['lwir', 'visible']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['lwir', 'visible']}
