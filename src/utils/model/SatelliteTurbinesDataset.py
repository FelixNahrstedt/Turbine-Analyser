import cv2
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import models

from matplotlib import image
import torch.nn.functional as F


class ImageDataset(Dataset):
    def __init__(self,csv, img_folder,transform):
        self.csv = csv
        self.transform = transform
        self.img_folder = img_folder
        self.bands = ["B2","B3","B4"]
        #get all Data with unspin


        self.image_names = self.csv[:]['id']
        self.dates = self.csv[:]['date']
        self.quality = self.csv[:]['quality']
        self.labels=self.csv[:]['label']
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.image_names)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        #Select Sample
        imgArr = []
        for band in self.bands:
            path = f'{self.img_folder}/{self.image_names[index]}-{self.dates[index]}-{band}.jpg'
            imgArr.append(image.imread(path))

        stacked = (np.dstack((imgArr[0],imgArr[1],imgArr[2]))).astype(np.uint8)

        item=self.transform(stacked)
        target=int(self.labels[index])
        if(target==2):
            target = 1
        return item, target 

class Net(nn.Module):
    def __init__(self, n_chans1 = 20):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_chans1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1//2, kernel_size=3, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_chans1//2)
        self.fc1 = nn.Linear((n_chans1//2) * (5//2) *(5//2), 50)
        self.fc2 = nn.Linear(50, 2)
    def forward(self, x):
        out = self.conv1_batchnorm(self.conv1(x))
        out = F.max_pool2d(torch.tanh(out), 2)
        out = self.conv2_batchnorm(self.conv2(out))
        out = F.max_pool2d(torch.tanh(out), 2)
        out = out.view(-1, (self.n_chans1//2) * (5//2) *(5//2))
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out
