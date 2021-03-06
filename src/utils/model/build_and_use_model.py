import csv
import random
import numpy as np
import pandas as pd
import torch
from SatelliteTurbinesDataset import Net,BaseModel
from training import validateHeights
from training import training_loop
from training import validate
from torchvision import models
from torch.utils.data import DataLoader
from SatelliteTurbinesDataset import ImageDataset
import torchvision.transforms as transforms
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/SW-Model-Turbines')
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


import torch.nn as nn
path_data = 'Data'
turbine_data = path_data +"/data_science/CSV/Sentinel-2-WindTurbineData.csv"
path_jpg = f'{path_data}/data_science/AllImages'
path_unspinned = f'{path_data}/data_science/not_spinning_images'
csvPath = path_data + "/data_science/CSV/"
path_gif = f'{path_data}/Satellite/GIF'


# TestSet = path_data + "/data_science/CSV/TestSet.csv"
# TrainSet = path_data + "/data_science/CSV/TrainSet.csv"
#TrainSet = "Data\data_science\CSV\TrainSet.csv"
newTrain = "Data/data_science/CSV/NewUndetectedComparison/train-2000.csv"
newTest = "Data/data_science/CSV/NewUndetectedComparison/test-2000.csv"
#TestSet = "Data\data_science\CSV\TestSet.csv"
#sort 1 csv into 3 csvs

path_undetected = "Data/data_science/No_Turbines_images"
#display_data(turbine_data)
# go through the good quality Data and convert it into still wind turbines
#
#sort images
seed = 69
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def train_and_validate(path_train_set,path_test_set,path_images,BATCH_SIZE, pretrainedModelPath = ""):
    torch.manual_seed(123)
    trainSet=pd.read_csv(path_train_set)
    testSet=pd.read_csv(path_test_set)
    heightSetTiny = pd.read_csv("Data/data_science/CSV/Tubine-Height-comparison/Turbines-With-Heights_tiny.csv")
    heightSetSmall = pd.read_csv("Data/data_science/CSV/Tubine-Height-comparison/Turbines-With-Heights_small.csv")
    heightSetMedium = pd.read_csv("Data/data_science/CSV/Tubine-Height-comparison/Turbines-With-Heights_medium.csv")
    heightSetLarge = pd.read_csv("Data/data_science/CSV/Tubine-Height-comparison/Turbines-With-Heights_large.csv")
    heightSetXLarge = pd.read_csv("Data/data_science/CSV/Tubine-Height-comparison/Turbines-With-Heights_xlarge.csv")

    
    #trainSet = drop_unspinned_data(trainSet_complete)
    #testSet = drop_unspinned_data(testSet_complete)


    #transform for pretrainedNets
    # transform =transforms.Compose([transforms.ToPILImage(),
    #                 transforms.Resize(255),
    #                 transforms.CenterCrop(224),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
    #             )
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((40,40)),
        transforms.ToTensor()
        ])
    train_dataset=ImageDataset(trainSet,path_images,transform)
    val_dataset=ImageDataset(testSet,path_images,transform)
    #height datasets
    tiny_set = ImageDataset(heightSetTiny,path_images,transform)
    small_set = ImageDataset(heightSetSmall,path_images,transform)
    medium_set= ImageDataset(heightSetMedium,path_images,transform)
    large_set = ImageDataset(heightSetLarge,path_images,transform)
    xLarge_set = ImageDataset(heightSetXLarge,path_images,transform)

    

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Training on device {device}.")


    print(train_dataset.__len__())
    print(val_dataset.__len__())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    #Height DataLoaders:
    tiny = torch.utils.data.DataLoader(tiny_set,BATCH_SIZE, False)
    small = torch.utils.data.DataLoader(small_set,BATCH_SIZE, False)
    medium = torch.utils.data.DataLoader(medium_set,BATCH_SIZE, False)
    large  = torch.utils.data.DataLoader(large_set,BATCH_SIZE, False)
    xLarge = torch.utils.data.DataLoader(xLarge_set,BATCH_SIZE, False)

    #Loading 

   # Old Classifier: 
    # model = models.densenet121(pretrained=True)
    # model.classifier = nn.Sequential(nn.Linear(1024,512),
    # nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256,2))
    # # print(model.classifier)
    # model.to(device=device)
    # if(pretrainedModelPath != ""):
    #     model.load_state_dict(torch.load(pretrainedModelPath))

    #Training
    depth = 7
    i = -1
    # Bisektions Verfahren von Funktion in region [0.1, 0.00001]
    momentum = [0, 0.99]
    lossAccBisec =[0,0]
    n_epochs = 16
    lr =0.0430908203125
    model = Net().to(device = device)
    model.load_state_dict(torch.load("Data/data_science/Models/Final-Model.pt"))
    optimizer = optim.SGD(model.parameters(), lr=lr) 
    loss_fn = nn.CrossEntropyLoss()
    name = "Final-Model"
    


    summary(model,input_size=(3,40,40))
    images,labels = next(iter(val_loader))
    print(images.shape)
    writer=SummaryWriter('/runs/logsdir')

    writer.add_graph(model,images.to(device="cuda"))
    print(f"parameters: {str(sum(p.numel() for p in model.parameters()))}")
    # bestEpoch, bestLoss, bestAccuracy = training_loop(
    # n_epochs = n_epochs,
    # optimizer = optimizer,
    # model = model,
    # loss_fn = loss_fn,
    # train_loader = train_loader,
    # device=device,
    # name=name,
    # val=val_loader,
    # path_save=f"Data/data_science/CSV/Model Improvements/improvements.csv"
    # )
    #print(f"best_epoch: {bestEpoch}, best_loss: {bestLoss}, bestAccuracy: {bestAccuracy}")
    
    #torch.save(model.state_dict(),path_data+f"/data_science/Models/{name}.pt")
    validateHeights(model,tiny,small,medium,large,xLarge,device)
    validate(model, train_loader, val_loader, device=device)
    

def drop_unspinned_data(df:pd.DataFrame):
    ids = df['id']
    length = ids.size
    idList = []
    for i in range(length):
        id = ids[i]
        if("unspin" in id):
            idList.append(i)
        
    data = df.drop(labels=idList, axis=0).reset_index(drop=True)
    return data

def select_n_random(data, labels, n=100):
        '''
        Selects n random datapoints and their corresponding labels from a dataset
        '''  
        assert len(data) == len(labels)

        perm = torch.randperm(len(data))
        return data[perm][:n], labels[perm][:n]
        
train_and_validate(newTrain,newTest,path_jpg,8, path_data+"/data_science/Models/Sentinel_2_pretrained-Dense.pt")


