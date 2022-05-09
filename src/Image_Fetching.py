from utils.data_collection.convert import convert_img

from utils.data_collection.download_Satellite_tiff import image_preparation
from utils.data_labeling.empty_Data import deleteImages
from utils.data_labeling.load_data import import_json

#Constants
path_data = 'C:/Users/fe-na/OneDrive/Dokumente/0 - Meine Dateien/Umweltinformatik/Eigene Projekte/Machine Learning/pytorch/sentinel-2-bewegungserkennung/Data'
path_tiffs = f'{path_data}/Tiffs'
path_jpg = f'{path_data}/Satellite/JPG'
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
# Main Loop for Sentinel-2 Image Downloads for all Counties
#
# (C) 2022 Felix Nahrstedt, Berlin, Germany
# email contact@felixnahrstedt.com
# -----------------------------------------------------------
def fetchImages(progress_csv,bundesLänder,dateImg,bands,img_path, tiff_path,basePath):
    result = []
    with open(progress_csv, "r") as f1:
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
            deleteImages(f"{basePath}/Satellite")
            startDate, endDate = image_preparation(tiff_path,img_path, bands,data['latitude'],data['longitude'],dateImg,converter)
            if(startDate == False):
                continue
            print(data['latitude'],data['longitude'])
            print(f'\nBundesland: {bundesLänder[bundesland]}, Windrad: {windrad},\n ')
            
            
            windrad = windrad+1
            stop = input("Wanna break = stop")
            if(stop == "stop"):
                break
        bundesland += 1

fetchImages(loadProgress,bundesLänder,'2022-03-14',['B2','B3','B4'],path_jpg,path_tiffs,path_data)

