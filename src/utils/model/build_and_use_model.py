import pandas as pd
import torch
from training import validate
from SatelliteTurbinesDataset import ImageDataset, Net
import torchvision.transforms as transforms


path_data = 'C:/Users/fe-na/OneDrive/Dokumente/0 - Meine Dateien/Umweltinformatik/Eigene Projekte/Machine Learning/pytorch/sentinel-2-bewegungserkennung/Data'
turbine_data = path_data +"/data_science/CSV/Sentinel-2-WindTurbineData.csv"
path_jpg = f'{path_data}/data_science/img_database'
path_unspinned = f'{path_data}/data_science/not_spinning_images'
csvPath = path_data + "/data_science/CSV/"
path_gif = f'{path_data}/Satellite/GIF'


TestSet = path_data + "/data_science/CSV/TestSet.csv"
TrainSet = path_data + "/data_science/CSV/TrainSet.csv"
#sort 1 csv into 3 csvs


#display_data(turbine_data)
# go through the good quality Data and convert it into still wind turbines
#
#sort images

def train_and_validate(path_train_set,path_test_set,path_images,BATCH_SIZE, pretrainedModelPath = ""):

    trainSet_complete=pd.read_csv(path_train_set)
    testSet_complete=pd.read_csv(path_test_set)
    
    trainSet = drop_unspinned_data(trainSet_complete)
    testSet = drop_unspinned_data(testSet_complete)



    transform =transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize((40, 40)),
                    transforms.ToTensor()],
                )
    
    train_dataset=ImageDataset(trainSet,path_images,transform)
    val_dataset=ImageDataset(testSet,path_images,transform)

    device = (torch.device('cuda') if torch.cuda.is_available()
    else torch.device('cpu'))
    print(f"Training on device {device}.")

    print(train_dataset.__len__())
    print(val_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
    shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE,
    shuffle=False)
    #Loading 
    model = Net().to(device = device) 
    model.load_state_dict(torch.load(pretrainedModelPath))



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

train_and_validate(TrainSet,TestSet,path_jpg,64,(path_data+"/data_science/Models/Sentinel_2.pt"))
