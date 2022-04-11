from cv2 import waitKey
import cv2
import imageio
import pandas as pd
import torch
from torch import nn
from utils.training import training_loop, validate
from utils.SatelliteTurbinesDataset import ImageDataset, Net
from utils.data_information import createSubCsv, display_data, splitTrainTest, unspin_turbines
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import datetime
import torch.optim as optim


path_data = 'C:/Users/fe-na/OneDrive/Dokumente/0 - Meine Dateien/Umweltinformatik/Eigene Projekte/Machine Learning/pytorch/sentinel-2-bewegungserkennung/Data'
turbine_data = path_data +"/data_science/CSV/Sentinel-2-WindTurbineData.csv"
path_jpg = f'{path_data}/data_science/img_database'
path_unspinned = f'{path_data}/data_science/not_spinning_images'
csvPath = path_data + "/data_science/CSV/"
path_gif = f'{path_data}/Satellite/GIF'

spinningCsv = path_data + "/data_science/CSV/spinning.csv"
notSpinningCsv = path_data + "/data_science/CSV/not_spinning.csv"
UndetectedCsv = path_data + "/data_science/CSV/undetected.csv"
TestSet = path_data + "/data_science/CSV/TestSet.csv"
TrainSet = path_data + "/data_science/CSV/TrainSet.csv"
BATCH_SIZE = 64
#sort 1 csv into 3 csvs
classNames = ["spinning", "not spinning", "undetectable"]


#display_data(turbine_data)
# go through the good quality Data and convert it into still wind turbines
#
#sort images

#unspin_turbines(turbine_data,path_jpg,path_gif,path_jpg)
#createSubCsv(turbine_data,csvPath)
#split into training and testing
#splitTrainTest(spinningCsv, notSpinningCsv,UndetectedCsv,TrainSet,TestSet)
trainSet=pd.read_csv(TrainSet)
testSet=pd.read_csv(TestSet)
transform =transforms.Compose([transforms.ToPILImage(),
                transforms.Resize((40, 40)),
                transforms.ToTensor()],
               )
train_dataset=ImageDataset(trainSet,path_jpg,transform)
val_dataset=ImageDataset(testSet,path_jpg,transform)


# img = test_dataset.__getitem__(0)["image"]
# label = test_dataset.__getitem__(0)["label"]
# print(img.shape, img.dtype)
# print(img.min(), img.max())

# data = 255 * img.numpy() # Now scale by 255
# img = data.astype(np.uint8)
# imageio.mimsave(f'{csvPath}permuted.gif',img )

# output = conv(img.unsqueeze(0))
# print(img.unsqueeze(0).shape, output.shape)

device = (torch.device('cuda') if torch.cuda.is_available()
else torch.device('cpu'))
print(f"Training on device {device}.")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64,
shuffle=True)
#Loading 
model = Net().to(device = device) 
model.load_state_dict(torch.load(path_data+"/data_science/Models/Sentinel_2.pt"))



#Training

# model = Net().to(device = device) #
# optimizer = optim.SGD(model.parameters(), lr=1e-2) #
# loss_fn = nn.CrossEntropyLoss()
# print(sum(p.numel() for p in model.parameters()))
# training_loop(
# n_epochs = 100,
# optimizer = optimizer,
# model = model,
# loss_fn = loss_fn,
# train_loader = train_loader,
# device=device
# )
# torch.save(model.state_dict(),path_data+"/data_science/Models/Sentinel_2.pt")
print(len(train_loader))
print(train_loader)
validate(model, train_loader, val_loader, device=device)

