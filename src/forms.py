from flask_wtf import FlaskForm
import pandas as pd
from parso import split_lines
from wtforms import (DecimalField, TextAreaField, IntegerField, BooleanField,
                     RadioField,DateField)
import matplotlib.pyplot as plt
from wtforms.validators import InputRequired, Length
from wtforms.validators import NumberRange

from utils.data_preperation.data_information import display_data

class LatLongForm(FlaskForm):
    latitude = DecimalField('latitude',render_kw={'class':'form-control'}, validators=[InputRequired(),
                                             NumberRange(min=-90, max=90, message='The numbers are in decimal degrees format and range from -90 to 90.')])
    longitude = DecimalField('longitude',render_kw={'class':'form-control'}, validators=[InputRequired(),
                                             NumberRange(min=-180, max=180, message='The numbers are in decimal degrees format and range from -180 to 180.')])
    datumField = DateField('Datum',render_kw={'class':'form-control'}, validators=[InputRequired()])


# def plotLineForThesis():
#     path = "Data/data_science/CSV/baseModel.csv"
#     size = str(250)
#     df = pd.read_csv(path)
#     df = df[df["name"]=="Base-Model-Deep2"]
#     lines1 = df.plot.line(x='epoch', y=['train_acc', 'val_acc', "train_loss","val_loss"])
#     #lines2 = df2.plot.line(x='epoch', y=['train_acc', 'val_acc', "train_loss","val_loss"])
#     plt.title('Loss and Accuracy with a deeper Base Model')
#     plt.ylabel('Accuracy/Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['training accuracy', 'validation accuracy',"training loss", "validation loss"], loc='lower right')
#     plt.savefig("Data/data_science/PLOTS/BaseModel-Deep.png")

#     plt.show()

# def plotBarForThesis():
#     sizes = [0,5,10,15,20]
#     path = "Data/data_science/CSV/Turbine-Model-depth-comparison/"
#     df = pd.read_csv(path+f"baseModel-depth.csv")
#     finalDf:pd.DataFrame = pd.DataFrame(columns=df.columns.values.tolist(), )
#     myList = []
#     for num in sizes:
#         mydf = df[df["name"]==f"Model-Depth-{num}"]
#         myList.append(df.iloc[[mydf["val_acc"].idxmax()]])
#     for i in myList:
#         finalDf = pd.concat([finalDf,i])
#     dfBar = pd.DataFrame({'train_acc': finalDf["train_acc"].values,
#                    'train_loss': finalDf["train_loss"].values,'val_acc': finalDf["val_acc"].values,'val_loss': finalDf["val_loss"].values}, index=sizes)
#     axes = dfBar.plot.bar(rot=0)
#     plt.subplots_adjust(hspace = 1)
#     plt.title('Model Depth Comparison')
#     plt.ylabel('Accuracy/Loss')
#     plt.xlabel('Size Training')
#     plt.savefig("Data/data_science/PLOTS/BaseModel-depth.png")
    
#     print(df)
#     df1 =  df[df["name"]==f"Model-Depth-{20}"]
#     df2 = df[df["name"]=="Model-Depth-0"]
#     lines1 = df1.plot.line(x='epoch', y=['train_acc', 'val_acc', "train_loss","val_loss"])
#     plt.title('Simple Model')
#     plt.ylabel('Accuracy/Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['training accuracy', 'validation accuracy',"training loss", "validation loss"], loc='lower right')
#     plt.savefig("Data/data_science/PLOTS/Shallow.png")
#     lines2 = df2.plot.line(x='epoch', y=['train_acc', 'val_acc', "train_loss","val_loss"])
#     plt.title('Deeper Model')
#     plt.ylabel('Accuracy/Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['training accuracy', 'validation accuracy',"training loss", "validation loss"], loc='lower right')
#     plt.savefig("Data/data_science/PLOTS/Deep.png")
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