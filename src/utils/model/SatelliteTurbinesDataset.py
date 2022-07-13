from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
import numpy as np
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
        item:torch.Tensor = self.transform(stacked)
        
        target = int(self.labels[index])
        if(target==2):
            target = 1
        return item, target

class BaseModel(nn.Module):
    def __init__(self, features = 20):
        super().__init__()
        self.features = features
        self.conv1 = nn.Conv2d(3, features, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=features)
        self.conv2 = nn.Conv2d(features, features//2, kernel_size=3, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=features//2)
        self.fc1 = nn.Linear((features//2) * (10) *(10), 50)
        self.fc2 = nn.Linear(50, 2)
    def forward(self, x):
        out = self.conv1_batchnorm(self.conv1(x))
        out = F.max_pool2d(torch.tanh(out), 2)
        out = self.conv2_batchnorm(self.conv2(out))
        out = F.max_pool2d(torch.tanh(out), 2)
        out = out.view(-1, (self.features//2) * (10) *(10))
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out
    

class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=1, bias=False)
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
    def __init__(self, features=30, n_blocks=5):
        super().__init__()
        self.features = features
        self.conv1 = nn.Conv2d(3, features, kernel_size=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=features)
        self.conv2 = nn.Conv2d(features, features//2, kernel_size=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=features//2)
        self.resblocks = nn.Sequential(
        *(n_blocks * [ResBlock(n_chans=features//2)]))
        self.fc1 = nn.Linear((10) *(10)* (features//2), 32)
        self.fc2 = nn.Linear(32, 2)
    def forward(self, x):
        out = self.conv1_batchnorm(self.conv1(x))
        out = F.max_pool2d(torch.tanh(out),2)
        out = self.conv2_batchnorm(self.conv2(out))
        out = self.resblocks(out)
        out = F.max_pool2d(torch.tanh(out), 2)
        #out1 =  F.max_pool2d(out1, 2)
        out = out.view(-1, (10) *(10)*  (self.features//2))
        #out1 = out1.view(-1, (10) *(10)*  (self.features//2))
        #newout = torch.cat((out),1)
        out = torch.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out