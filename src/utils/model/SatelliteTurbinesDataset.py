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
from torchvision.transforms import functional
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
        ImgCropTransform = T.Compose([T.ToPILImage(),T.Resize((40,40)),T.ToTensor()])
        for band in self.bands:
            path = f'{self.img_folder}/{self.image_names[index]}-{self.dates[index]}-{band}.jpg'
            imgArr.append(image.imread(path))

        stacked = (np.dstack((imgArr[0],imgArr[1],imgArr[2]))).astype(np.uint8)
        stacked = ImgCropTransform(stacked)
        stacked = stacked[:,20:40,10:30]
        item = self.transform(stacked)
        
        target = int(self.labels[index])
        
        if(target==2):
            target = 1
        return item, target

# class Net(nn.Module):
#     def __init__(self, n_chans1 = 20):
#         super().__init__()
#         self.n_chans1 = n_chans1
#         self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
#         self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_chans1)
#         self.conv2 = nn.Conv2d(n_chans1, n_chans1//2, kernel_size=3, padding=1)
#         self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_chans1//2)
#         self.fc1 = nn.Linear((n_chans1//2) * (20//2) *(20//2), 50)
#         self.fc2 = nn.Linear(50, 2)
#     def forward(self, x):
#         out = self.conv1_batchnorm(self.conv1(x))
#         out = F.max_pool2d(torch.tanh(out), 2)
#         out = self.conv2_batchnorm(self.conv2(out))
#         out = F.max_pool2d(torch.tanh(out), 2)
#         out = out.view(-1, (self.n_chans1//2) * (20//2) *(20//2))
#         out = torch.tanh(self.fc1(out))
#         out = self.fc2(out)
#         return out

class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=3,
                                padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)
        torch.nn.init.kaiming_normal_(self.conv.weight,
        nonlinearity='relu')
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)
        torch.nn.init.zeros_(self.batch_norm.bias)
    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x

class Net(nn.Module):
    #chans = 32
    #chans less features = 12
    #blocks = 3
    #more blocks 12
    def __init__(self, n_chans1=3, n_blocks=2):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=n_chans1)
        self.conv2 = nn.Conv2d(n_chans1, n_chans1//2, kernel_size=3, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=n_chans1//2)
        self.resblocks = nn.Sequential(
        *(n_blocks * [ResBlock(n_chans=n_chans1//2)]))
        self.fc1 = nn.Linear((10//2) *(10//2)* (n_chans1//2), 32)
        self.fc2 = nn.Linear(32, 2)
    def forward(self, x):
        out = self.conv1_batchnorm(self.conv1(x))
        out = F.max_pool2d(torch.tanh(out), 2)
        out = self.conv2_batchnorm(self.conv2(out))
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(-1, (10//2) *(10//2)*  (self.n_chans1//2))
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out