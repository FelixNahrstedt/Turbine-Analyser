from math import sqrt
from flask_wtf import FlaskForm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from wtforms import (DecimalField,DateField)
from wtforms.validators import InputRequired
from wtforms.validators import NumberRange
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from utils.model.SatelliteTurbinesDataset import Net
from utils.data_preperation.data_information import display_data

class LatLongForm(FlaskForm):
    latitude = DecimalField('latitude',render_kw={'class':'form-control'}, validators=[InputRequired(),
                                             NumberRange(min=-90, max=90, message='The numbers are in decimal degrees format and range from -90 to 90.')])
    longitude = DecimalField('longitude',render_kw={'class':'form-control'}, validators=[InputRequired(),
                                             NumberRange(min=-180, max=180, message='The numbers are in decimal degrees format and range from -180 to 180.')])
    datumField = DateField('Datum',render_kw={'class':'form-control'}, validators=[InputRequired()])
# def train(lr):
#     return [0.05+sqrt(lr), 90+sqrt(lr)]

# def plotLineForThesis():
#     path = "Data/data_science/CSV/Tubine-Height-comparison/Turbines-With-Heights.csv"

#     df = pd.read_csv(path)
#     name = "turbine_data_Height"
#     df = df["heightInM"]
#     print(df.mean())
#     i = 0
#     lengths = [0,0,0,0,0]
#     blub = 34
#     while i<len(df):
#         a = df.iloc[i]
#         print(a)
#         if(a<=blub):
#             lengths[0] +=1/len(df)
#         elif(a<=blub*2 and a>blub):
#             lengths[1] +=1/len(df)
#         elif(a<=blub*3 and a>blub*2):
#             lengths[2] +=1/len(df)
#         elif(a<=blub*4 and a>blub*3):
#             lengths[3] +=1/len(df)
#         elif(a<=blub*5 and a>blub*4):
#             lengths[4] +=1/len(df)
#         i = i+1
#     print(lengths)

    # fig = plt.figure(figsize = (10, 5))
    

    # langs = ['0m-34m', '35m-68m', '69-102', '103-136', '137-170']

    # plt.bar(langs,lengths,color="maroon",width = 0.4)
    # plt.xlabel("Turbine Height", fontsize = 14)
    # plt.ylabel("Turbine quantity in %", fontsize = 14)
    # plt.savefig(f"Data/data_science/PLOTS/{name}.png")

    # plt.show()
    # fig, ax = plt.subplots()

    # bar1 = ax.bar([1,2,3,4,5], df1['quality'], color="maroon",label = 'img')
    # ax.set_xlabel("Quality", fontsize = 14)

    # ax.set_ylabel("Images per class [img]",fontsize=14)
    # ax2=ax.twinx()

    # line1 = ax2.plot([1,2,3,4,5],df1['max_std_brightness'], color="goldenrod",linewidth=4, label="diff")
    # ax2.set_xlabel("Quality", fontsize = 14)
    # ax2.set_ylabel("Brightness differences [diff]",
    #           fontsize=14)
    

    # fig.savefig(f"Data/data_science/PLOTS/{name}.png")

    # lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()

    # ax2.legend(lines + lines2, labels + labels2, loc=0)


    #plt.savefig(f"Data/data_science/PLOTS/{name}.png")
    #plt.show()


    #df2 = pd.DataFrame(myList, columns=list([1,2,3,4,5]), index=['x', 'y'])
    # print(df2)
    # ax2 = df2.plot.line(rot=0,color ='g')
    # ax.set_xlabel('Quality')
    # ax.set_ylabel('Images per class', color='g')
    # ax2.set_ylabel('Maximum mean brightness', color='b')
    # plt.savefig(f"Data/data_science/PLOTS/{name}.png")
    # plt.show()

    # lines1 = df.plot.line(x='Step', y=['train', 'val'])
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(["training loss", "validation loss"], loc='center right')
    
#plotLineForThesis()
#     plt.savefig(f"Data/data_science/PLOTS/{name}.png",figsize=(2,5))
#     plt.show()
# plotLineForThesis()
#def plotBarForThesis():
    # sizes = [5,10,15,20,25,30,35,40,45,50]
    # siz = [ '5', '10', '15',"20","25","30","35","40","45","50"]
    # path = "Data/data_science/CSV/Model Improvements/"
    # df = pd.read_csv(path+f"improvements.csv")
    # finalDf:pd.DataFrame = pd.DataFrame(columns=df.columns.values.tolist(), )
    # myList = []
    # for num in sizes:
    #     mydf = df[df["name"]==f"ResWide-New-{num}"]
    #     myList.append(mydf["val_loss"].iloc[10])
    
    # fig = plt.figure(figsize = (5, 2))
 
    # # creating the bar plot
    # plt.bar(siz, myList, color ='maroon',
    #         width = 0.4)
    
    # plt.xlabel("extra residual layer")
    # plt.ylabel("Loss")
    # plt.show()
    #sizes = [5,10,15,20,25,30,35,40,45,50]
    # sizes = [1,5,10,15]
    # siz = ['1', '5', '10', '15']
    #siz = [ '5', '10', '15',"20","25","30","35","40","45","50"]
    # path = "Data/data_science/CSV/Model Improvements/"
    # df = pd.read_csv(path+f"improvements.csv")
    # finalDf:pd.DataFrame = pd.DataFrame(columns=df.columns.values.tolist(), )
    # myList = []
    # for num in sizes:
    #     mydf = df[df["name"]==f"ResDepth-New-{num}"]
    #     myList.append(mydf["val_loss"].iloc[10])
    
    # fig = plt.figure(figsize = (5, 2))
    # # creating the bar plot
    # plt.bar(siz, myList, color ='maroon',
    #         width = 0.4)
    # plt.xlabel("extra convolutional input features")
    # plt.ylabel("Loss")
    # fig.tight_layout()
    # fig.savefig("Data/data_science/PLOTS/BaseModel-Width.png")

    # plt.show() 
    
    # print(df)
    # df1 =  df[df["name"]==f"Model-Depth-{20}"]
    # df2 = df[df["name"]=="Model-Depth-0"]
    # lines1 = df1.plot.line(x='epoch', y=['train_acc', 'val_acc', "train_loss","val_loss"])
    # plt.title('Simple Model')
    # plt.ylabel('Accuracy/Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['training accuracy', 'validation accuracy',"training loss", "validation loss"], loc='lower right')
    # plt.savefig("Data/data_science/PLOTS/Shallow.png")
    # lines2 = df2.plot.line(x='epoch', y=['train_acc', 'val_acc', "train_loss","val_loss"])
    # plt.title('Deeper Model')
    # plt.ylabel('Accuracy/Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['training accuracy', 'validation accuracy',"training loss", "validation loss"], loc='lower right')
    # plt.savefig("Data/data_science/PLOTS/Deep.png")
#plotBarForThesis()

#display_data("C:/Users/fe-na/OneDrive/Dokumente/0 - Meine Dateien/Umweltinformatik/Eigene Projekte/Machine Learning/pytorch/sentinel-2-bewegungserkennung/Data/data_science/CSV/raw-Data/Sentinel-2-WindTurbineData.csv")
# df = pd.read_csv("C:/Users/fe-na/OneDrive/Dokumente/0 - Meine Dateien/Umweltinformatik/Eigene Projekte/Machine Learning/pytorch/sentinel-2-bewegungserkennung/Data/data_science/CSV/raw-Data/Sentinel-2-WindTurbineData.csv")
# df2 = pd.read_csv("Data/data_science/CSV/NewUndetectedComparison/spinningAndUndetectedNew-2000.csv")
# list1 = df["id"].tolist()
# list2 = df2["id"].tolist()
# eighters = []
# for id in list1:
#     if(len(str(id))==8):
#         eighters.append(id)
# print(len(eighters))
# for id in eighters:
#     if id in list2:
#         print("help")
#     else:
#         print("no")

# def shortenCsvData():
#     df = pd.read_csv("Data/data_science/CSV/raw-Data/wind-farm-1-signals-testing.csv", sep=";")
#     turbines = [""]
#     df = df[df["Gen_RPM_Max"]==0]
#     turbines = set(df["Turbine_ID"])
#     li = sorted(turbines)

#     for turbine in turbines:
#         a = df[df["Turbine_ID"]==turbine]
#         times = a["Timestamp"]
#         times = set(times)
#         times = sorted(times)
#         newTimes = []
#         doubles = []
#         for time in times:
#             d = datetime.fromisoformat(time).astimezone(timezone.utc)
#             d = d.strftime('%Y-%m-%d')
#             if(time in newTimes):
#                 doubles.append(d)
#             else:
#                 newTimes.append(d)
#         print(str(turbine) +": "+ str(len(a))+" Amount of Timestamps: "+str(len(times)))
#         print("Failure data from the same day: " +str(len(doubles)))
#         print("Failure data not from the same day: " +str(len(newTimes)))

#     #df = df[df["Gen_RPM_Max"]==0]
#     print(len(df))
#     print(li)

# shortenCsvData()