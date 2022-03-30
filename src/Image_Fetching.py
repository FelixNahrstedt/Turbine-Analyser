from pathlib import Path
from datetime import date, datetime, timedelta
from utils.plots import *
from utils.data_information import appendCsv, appendCsv_open, createCsv
from utils.convert import *
from utils.data_information import evaluate_images
from utils.download_Satellite_tiff import image_preparation
from utils.empty_Data import deleteImages
from utils.load_data import import_json
import imquality.brisque as brisque
import datetime
import PIL.Image

from utils.change_detection import dense_optical_flow, lucas_kanade

#Constants
path_data = 'C:/Users/fe-na/OneDrive/Dokumente/0 - Meine Dateien/Umweltinformatik/Eigene Projekte/Machine Learning/pytorch/sentinel-2-bewegungserkennung/Data'
path_tiffs = f'{path_data}/Satellite/Tiffs'
path_jpg = f'{path_data}/data_science/img_database'
path_gif = f'{path_data}/Satellite/GIF'
path_locations = f'{path_data}/locations'
bundesLänder =  ["Bremen", "Schleswig-Holstein",
"Niedersachsen", "Sachsen", "Sachsen-Anhalt", "Hamburg",
"Berlin", "Brandenburg", "Nordrhein-Westfalen",
"Mecklenburg-Vorpommern", "Hessen", "Rheinland-Pfalz",
"Thüringen", "Saarland", "Bayern", "Baden-Württemberg"]
loadProgress = path_data + "/data_science/CSV/zwischenspeicher.csv"
bands = ['B2','B3','B4']
dateImg = '2022-03-14'
breakOutOfMatrix = False

# -----------------------------------------------------------
# Main Loop
#
# (C) 2022 Felix Nahrstedt, Berlin, Germany
# email contact@felixnahrstedt.com
# -----------------------------------------------------------
result = []
with open(loadProgress, "r") as f1:
    last_progress = f1.readlines()
    result = [x.strip() for x in last_progress[len(last_progress)-2].split(',')]

print(result)
bundesland = int(result[1])
loadWindrad_once = int(result[2])

while(bundesland < len(bundesLänder)):
    #geo locations
    jsonData, turbine_keys = import_json(bundesLänder[bundesland])
    #Sorting Loop
    windrad = 0 + loadWindrad_once
    loadWindrad_once = 0
    #dense_optical_flow(f'{path_gif}/1882031797-2022-03-14-HIST.gif', path_gif)
    while(windrad<len(turbine_keys)):
        key = turbine_keys[windrad]
        data =  jsonData[key]
        converter = convert_img(key,dateImg,bands)
        deleteImages(f"{path_data}/SatellitASe")
        image_preparation(path_tiffs,path_jpg, bands,data['latitude'],data['longitude'],dateImg,converter)
        
        print(data['latitude'],data['longitude'])
        print(f'\nBundesland: {bundesLänder[bundesland]}, Windrad: {windrad},\n ')
        #print(f'Maximum Mean Brightness: {maxMeanBrightness}') 
        #print(f'Maximum STD Brightness: {maxStdBrightness}')


        # enhancement

        # #imagePlot = img_plots(path_jpg, key, bands, dateImg)
        # #imgArrHist,matchArr = imagePlot.plot_img_distribution()
        # #converter = convert_img(key,dateImg,bands,add="-Histed")
        # #converter.saveToGif(path_gif,imgArrHist,"-HIST")
        # #converter.saveToGif(path_gif, localImgEnhanceArr, "-local")
        # #converter.saveToGif(path_gif, matchArr, "-Match")
        # #converter.saveToGif(path_gif, modernArt, "-sob")

        # # differences = imagePlot.motion_detector_Pixel_Differences("-CLAHE")
        # # converter.saveToGif(path_gif,differences, "-diff")
        # # lucas_kanade(f'{path_gif}/{key}-{dateImg}-HIST.gif',path_jpg)
        # # gif not recognizable
        # # createCsv(f'{path_data}/data_science/Sentinel-2-WindTurbineData.csv')
        # # appendCsv(f'{path_data}/data_science/not_recognizable.csv',key,data['latitude'],data['longitude'],maxStdBrightness,maxMeanBrightness)
        # # recognizable gif

        #createCsv(f'{path_data}/data_science/img_database/Sentinel-2-WindTurbineData.csv',['id', 'latitude','longitude','label','max_mean-bright', 'max_std_bright', 'date'])

        
        # inputIncorrect = True
        # while(inputIncorrect):
        #     recognizable = " "
        #     recognizable = input("Dreht sich das Windrad (d), Nicht (n), undefiniert (u): ")
        #     if(recognizable == "d"):
        #         inputIncorrect = False
        #         appendCsv(pathCsv=f'{path_data}/data_science/CSV/Sentinel-2-WindTurbineData.csv',id=key,lat=data['latitude'],lon=data['longitude'],mean=maxMeanBrightness,std=maxStdBrightness,date=dateImg, label=0)

        #     elif(recognizable== "n"):
        #         inputIncorrect = False
        #         appendCsv(pathCsv=f'{path_data}/data_science/CSV/Sentinel-2-WindTurbineData.csv',id=key,lat=data['latitude'],lon=data['longitude'],mean=maxMeanBrightness,std=maxStdBrightness,date=dateImg, label=1)

        #     elif(recognizable== "u"):
        #         inputIncorrect = False
        #         appendCsv(pathCsv=f'{path_data}/data_science/CSV/Sentinel-2-WindTurbineData.csv',id=key,lat=data['latitude'],lon=data['longitude'],mean=maxMeanBrightness,std=maxStdBrightness,date=dateImg, label=2)
            
        #     elif(recognizable=="stop"):
        #         inputIncorrect = False
        #         print(f'Bundesländer durch = {bundesland}; Windräder durch = {windrad-1}')
        #         appendCsv_open(pathCsv=f'{path_data}/data_science/CSV/zwischenspeicher.csv',allData=[datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f%z'),bundesland,windrad])
        #         breakOutOfMatrix = True
        #         break
        #     else:
        #         print("Wrong input")

        # if(breakOutOfMatrix):
        #     break
        windrad = windrad+1

    ##Pgenerate Plots
    # converter = convert_img(7472118681,dateImg,bands)
    # converter.plot_to_better_jpg(path_tiffs)
    # if(breakOutOfMatrix):
    #     break
    bundesland += 1


    # csv_information = recognizability(f'{path_data}/data_science/CSV/recognizable.csv',f'{path_data}/data_science/CSV/not_recognizable.csv')
    # csv_information.cluster()
    # csv_information.recognizability()
        
