import csv
from datetime import datetime
import pandas as pd
from utils.data_collection.convert import convert_img

from utils.data_collection.download_Satellite_tiff import image_preparation
from utils.data_labeling.empty_Data import deleteImages
from utils.data_labeling.load_data import import_json

#Constants
path_data = 'C:/Users/fe-na/OneDrive/Dokumente/0 - Meine Dateien/Umweltinformatik/Eigene Projekte/Machine Learning/pytorch/sentinel-2-bewegungserkennung/Data'
path_tiffs = f'{path_data}/Tiffs'
path_jpg = f'Data/data_science/No_Turbines_images'
path_gif = f'{path_data}/Satellite/GIF'
loadProgress = path_data + "/data_science/CSV/zwischenspeicher.csv"
bands = ['B2','B3','B4']
dateImg = '2022-03-14'
breakOutOfMatrix = False

# -----------------------------------------------------------
# Main Loop for Sentinel-2 Image Downloads for all Counties
#
# (C) 2022 Felix Nahrstedt, Berlin, Germany
# email contact@felixnahrstedt.com
# -----------------------------------------------------------
def fetchImages(progress_csv,path_data:pd.DataFrame,dateImg,bands,img_path, tiff_path,basePath):
    df:pd.DataFrame = pd.read_csv(path_data)
    progress = pd.read_csv(progress_csv)
    print(progress)
    next = progress.tail(1)
    print(next)
    next = int(next.iloc[0]["row"])
    print(next)


    while(next<df.shape[0]):
        data = df.iloc[[next]]
        key = data.iloc[0]["id"]
        datum = data.iloc[0]["date"]
        converter = convert_img(key,datum,bands)
        deleteImages(f"{basePath}/Satellite")
        startDate, endDate = image_preparation(tiff_path,img_path, bands,data.iloc[0]['latitude'],data.iloc[0]['longitude'],datum,converter)
        print(data.iloc[0]['latitude'],data.iloc[0]['longitude'])
        print(f'\Stadt: {data.iloc[0]["region"]}, Bild Nummer: {next},\n ')
        
        
        next = next+1
        if(startDate == False):
            df = df.drop(labels=(next-1), axis=0)
            print("deleted "+data.iloc[0]["region"])
            continue
        #stop = input("Wanna break = stop: ")
        if((next % 50)==0):
            dateTimeObj = datetime.now()
            dateStr = dateTimeObj.strftime("%d-%b-%Y-%H-%M")
            print(dateStr)
            city = data.iloc[0]["region"]
            id = data.iloc[0]["id"]
            with open(progress_csv, 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                # write the data
                writer.writerow([dateStr,next,city,id])
            df.to_csv('Data/data_science/CSV/raw-Data/Cities-not-Turbines-Updated.csv', index=False)
fetchImages(loadProgress,"Data/data_science/CSV/raw-Data/Cities-not-Turbines.csv",'2022-03-14',['B2','B3','B4'],path_jpg,path_tiffs,path_data)

