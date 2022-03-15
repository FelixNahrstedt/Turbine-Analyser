from pathlib import Path
from datetime import date, timedelta
from utils.plots import plot_csv_data_recognizability
from utils.data_information import appendCsv, createCsv
from utils.convert import saveToGif
from utils.data_information import evaluate_images
from utils.download_Satellite_tiff import image_preparation
from utils.empty_Data import deleteImages
from utils.load_data import import_json

#Paths
path_data = 'C:/Users/fe-na/OneDrive/Dokumente/0 - Meine Dateien/Umweltinformatik/Eigene Projekte/Machine Learning/pytorch/sentinel-2-bewegungserkennung/Data'
path_tiffs = f'{path_data}/Satellite/Tiffs'
path_jpg = f'{path_data}/Satellite/JPG'
path_gif = f'{path_data}/Satellite/GIF'
path_locations = f'{path_data}/locations'

#geo locations
jsonData, turbine_keys = import_json()

#options
bands = ['B4', 'B3', 'B2']
dateImg = '2022-03-14'

# i = 0
# while(i<49):
#     key = turbine_keys[i]
#     data =  jsonData[key]

#     deleteImages(f"{path_data}/Satellite")
#     image_preparation(path_tiffs,path_jpg, bands,data['latitude'],data['longitude'],dateImg, key)
#     maxMeanBrightness, maxStdBrightness, imgArr = evaluate_images(path_jpg,key,dateImg,bands)
#     saveToGif(path_gif,key,dateImg,imgArr)
#     print(data['latitude'],data['longitude'])
#     print(f'Maximum Mean Brightness: {maxMeanBrightness}') 
#     print(f'Maximum STD Brightness: {maxStdBrightness}')

#     ##gif not recognizable
#     #createCsv(f'{path_data}/data_science/not_recognizable.csv')
#     #appendCsv(f'{path_data}/data_science/not_recognizable.csv',key,data['latitude'],data['longitude'],maxStdBrightness,maxMeanBrightness)

#     ## recognizable gif
#     # createCsv(f'{path_data}/data_science/recognizable.csv',['id', 'latitude','longitude', 'std_bright', 'mean_std'])
#     #appendCsv(f'{path_data}/data_science/recognizable.csv',key,data['latitude'],data['longitude'],maxStdBrightness,maxMeanBrightness)

#     recognizable = " "
#     recognizable = input("Is Wind Turbine Recognizable?")

#     if(recognizable == "1"):
#         appendCsv(f'{path_data}/data_science/recognizable.csv',key,data['latitude'],data['longitude'],maxStdBrightness,maxMeanBrightness)
#     elif(recognizable== "0"):
#         appendCsv(f'{path_data}/data_science/not_recognizable.csv',key,data['latitude'],data['longitude'],maxStdBrightness,maxMeanBrightness)
#     i = i+1

plot_csv_data_recognizability(f'{path_data}/data_science/CSV/recognizable.csv',f'{path_data}/data_science/CSV/not_recognizable.csv')
    
