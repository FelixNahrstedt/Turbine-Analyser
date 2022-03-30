
#constants:
import datetime
from utils.convert import convert_img
from utils.data_information import appendCsv, appendCsv_open, createCsv, evaluate_images
from utils.empty_Data import deleteGiffs
from utils.load_data import import_json


path_data = 'C:/Users/fe-na/OneDrive/Dokumente/0 - Meine Dateien/Umweltinformatik/Eigene Projekte/Machine Learning/pytorch/sentinel-2-bewegungserkennung/Data'
path_jpg = f'{path_data}/data_science/img_database'
path_locations = f'{path_data}/locations'
path_gif = f'{path_data}/Satellite/GIF'

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
# Main Loop for Labeling
#
# (C) 2022 Felix Nahrstedt, Berlin, Germany
# email contact@felixnahrstedt.com
# -----------------------------------------------------------

# Get all keys 

result = []
with open(loadProgress, "r") as f1:
    last_progress = f1.readlines()
    print(len(last_progress))
    if(len(last_progress)<=2):
        result = [x.strip() for x in last_progress[-1].split(',')]
    else:
        result = [x.strip() for x in last_progress[len(last_progress)-1].split(',')]

print(result)
bundesland = int(result[1])
loadWindrad_once = int(result[2])
# create Loop
while(bundesland < len(bundesLänder)):
    #geo locations
    jsonData, turbine_keys = import_json(bundesLänder[bundesland])
    #Sorting Loop
    windrad = 0 + loadWindrad_once
    loadWindrad_once = 0
    while(windrad<len(turbine_keys)):
        key = turbine_keys[windrad]
        data =  jsonData[key]
    # create large gif and show it 
        converter = convert_img(key,dateImg,bands)
        deleteGiffs(path_gif)
        print(data['latitude'],data['longitude'])
        print(f'Bundesland = {bundesLänder[bundesland]}; Windräder durch = {windrad}')

        maxMeanBrightness, maxStdBrightness, imgArr = evaluate_images(path_jpg,key,dateImg,bands)
        converter.saveToGif(path_gif,imgArr)

        # create CSV for storing Data
        ###createCsv(f'{path_data}/data_science/CSV/Sentinel-2-WindTurbineData.csv',['id', 'latitude','longitude','label','quality','max_mean-bright', 'max_std_bright', 'date','region'])
        
        # Ask for spin/no spin/ not recognizable and Image Quality
        print(data['latitude'],data['longitude'])
        inputIncorrect = True
        while(inputIncorrect):
            recognizable = " "
            recognizable = input("SPIN (s) - NO SPIN (n) - Or Unrecognizable (u) --- or 'stop': ")
            if(recognizable == "s"):
                #s for Spin
                inputIncorrect = False
                secInputCorrect = True
                while(secInputCorrect):
                    quality = input("Bewerten Sie die Erkennbarkeit 1(Sehr Gut)-5(Ich rate): ")
                    if(quality.isnumeric()):
                        appendCsv(
                            pathCsv=f'{path_data}/data_science/CSV/Sentinel-2-WindTurbineData.csv',
                            id=key,
                            lat=data['latitude'],
                            lon=data['longitude'],
                            mean=maxMeanBrightness,
                            std=maxStdBrightness,
                            date=dateImg, 
                            qual=quality,
                            region=bundesLänder[bundesland],
                            label=0)
                        secInputCorrect = False

            elif(recognizable== "n"):
                #n for no Spin
                inputIncorrect = False
                secInputCorrect = True
                while(secInputCorrect):
                    quality = input("Bewerten Sie die Erkennbarkeit 1(Sehr Gut)-5(Ich rate): ")
                    if(quality.isnumeric()):
                        appendCsv(
                            pathCsv=f'{path_data}/data_science/CSV/Sentinel-2-WindTurbineData.csv',
                            id=key,
                            lat=data['latitude'],
                            lon=data['longitude'],
                            mean=maxMeanBrightness,
                            std=maxStdBrightness,
                            date=dateImg, 
                            qual=quality,
                            region=bundesLänder[bundesland],
                            label=1)
                        secInputCorrect = False

            elif(recognizable== "u"):
                #u for uncrecognizable
                inputIncorrect = False
                appendCsv(
                            pathCsv=f'{path_data}/data_science/CSV/Sentinel-2-WindTurbineData.csv',
                            id=key,
                            lat=data['latitude'],
                            lon=data['longitude'],
                            mean=maxMeanBrightness,
                            std=maxStdBrightness,
                            date=dateImg, 
                            qual=6,
                            region=bundesLänder[bundesland],
                            label=2)
            
            elif(recognizable=="stop"):
                inputIncorrect = False
                print(f'Bundesländer durch = {bundesland}; Windräder durch = {windrad}')
                appendCsv_open(pathCsv=f'{path_data}/data_science/CSV/zwischenspeicher.csv',allData=[datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f%z'),bundesland,windrad])
                breakOutOfMatrix = True
                break
            else:
                print("Wrong input")

        if(breakOutOfMatrix):
            break
        else:
            windrad = windrad+1
    if(breakOutOfMatrix):
        break
    bundesland += 1

