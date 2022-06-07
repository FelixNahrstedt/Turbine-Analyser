
#constants:

from utils.data_collection.convert import convert_img
from utils.data_preperation.data_information import evaluate_images
from utils.data_labeling.empty_Data import deleteGiffs
from utils.data_labeling.load_data import import_json
from utils.data_collection.img_plots import img_plots


path_data = 'Data'
path_jpg = f'{path_data}/data_science/img_database'
#path_locations = f'{path_data}/locations'
path_gif = f'{path_data}/Satellite/GIF'

bundesLänder =  ["Bremen", "Schleswig-Holstein",
"Niedersachsen", "Sachsen", "Sachsen-Anhalt", "Hamburg",
"Berlin", "Brandenburg", "Nordrhein-Westfalen",
"Mecklenburg-Vorpommern", "Hessen", "Rheinland-Pfalz",
"Thüringen", "Saarland", "Bayern", "Baden-Württemberg"]
loadProgress = path_data + "/data_science/CSV/zwischenspeicher.csv"
turbine_data = path_data +"/data_science/CSV/Sentinel-2-WindTurbineData.csv"
# Datum vllt. mal 24.03?

# Get all keys 
def label_data(progress_csv, turbine_data_csv, bundesländer, path_gifs, path_images, date, bandsImg, basePath_csv):
    breakOutOfMatrix = False
    result = []
    allTurbinesSoFar = []
    with open(progress_csv, "r") as f1:
        last_progress = f1.readlines()
        if(len(last_progress)<=2):
            result = [x.strip() for x in last_progress[-1].split(',')]
        else:
            result = [x.strip() for x in last_progress[len(last_progress)-1].split(',')]

    # Check for Doubled elements in database
    with open(turbine_data_csv, "r") as f1:
        keysSoFar = f1.readlines()[1:]
    doneKeys = []
    for row in keysSoFar:
        doneKeys.append(int(row.split(',')[0]))

    print(f'Wind Turbines looked at so far: {len(doneKeys)}')
    bundesland = int(result[1])
    loadWindrad_once = int(result[2])
    # create Loop
    while(bundesland < len(bundesländer)):
        #geo locations
        jsonData, turbine_keys = import_json(bundesländer[bundesland])
        #Sorting Loop
        windrad = 0 + loadWindrad_once
        loadWindrad_once = 0
        while(windrad<len(turbine_keys)):
            key = turbine_keys[windrad]
            data =  jsonData[key]
        # create large gif and show it 
            converter = convert_img(key,date,bandsImg)
            deleteGiffs(path_gifs)
            print('\n')
            print(data['latitude'],data['longitude'])
            print(f'Bundesland = {bundesländer[bundesland]}; Windräder durch = {windrad}')

            maxMeanBrightness, maxStdBrightness, imgArr = evaluate_images(path_images,key,date,bandsImg)
            matchingImgs = img_plots(path_images,key,bandsImg,date)
            matchedArr = matchingImgs.histogram_matching(imgArr)
            converter.saveToGif(path_gifs,matchedArr,imgArr)

            #create CSV for storing Data
            ##createCsv(f'{path_data}/data_science/CSV/Sentinel-2-WindTurbineData.csv',['id', 'latitude','longitude','label','quality','max_mean-bright', 'max_std_bright', 'date','region'])
            
            #Ask for spin/no spin/ not recognizable and Image Quality

            #labeling -- enable to label
            # inputIncorrect = True
            # while(inputIncorrect):
            #     recognizable = " "
            #     recognizable = input("SPIN (s) - NO SPIN (n) - Or Unrecognizable (u) --- or 'stop': ")
            #     if(recognizable == "s"):
            #         #s for Spin
            #         inputIncorrect = False
            #         secInputCorrect = True
            #         while(secInputCorrect):
            #             quality = input("Bewerten Sie die Erkennbarkeit 1(Sehr Gut)-5(Ich rate): ")
            #             if(quality.isnumeric()):
            #                 appendCsv(
            #                     pathCsv=f'{basePath_csv}/data_science/CSV/Sentinel-2-WindTurbineData.csv',
            #                     id=key,
            #                     lat=data['latitude'],
            #                     lon=data['longitude'],
            #                     mean=maxMeanBrightness,
            #                     std=maxStdBrightness,
            #                     date=date, 
            #                     qual=quality,
            #                     region=bundesländer[bundesland],
            #                     label=0)
            #                 secInputCorrect = False

            #     elif(recognizable== "n"):
            #         #n for no Spin
            #         inputIncorrect = False
            #         secInputCorrect = True
            #         while(secInputCorrect):
            #             quality = input("Bewerten Sie die Erkennbarkeit 1(Sehr Gut)-5(Ich rate): ")
            #             if(quality.isnumeric()):
            #                 appendCsv(
            #                     pathCsv=f'{basePath_csv}/data_science/CSV/Sentinel-2-WindTurbineData.csv',
            #                     id=key,
            #                     lat=data['latitude'],
            #                     lon=data['longitude'],
            #                     mean=maxMeanBrightness,
            #                     std=maxStdBrightness,
            #                     date=date, 
            #                     qual=quality,
            #                     region=bundesländer[bundesland],
            #                     label=1)
            #                 secInputCorrect = False

            #     elif(recognizable== "u"):
            #         #u for uncrecognizable
            #         inputIncorrect = False
            #         appendCsv(
            #                     pathCsv=f'{basePath_csv}/data_science/CSV/Sentinel-2-WindTurbineData.csv',
            #                     id=key,
            #                     lat=data['latitude'],
            #                     lon=data['longitude'],
            #                     mean=maxMeanBrightness,
            #                     std=maxStdBrightness,
            #                     date=date, 
            #                     qual=6,
            #                     region=bundesländer[bundesland],
            #                     label=2)
                
            #     elif(recognizable=="stop"):
            #         inputIncorrect = False
            #         print(f'Bundesländer durch = {bundesland}; Windräder durch = {windrad}')
            #         appendCsv_open(pathCsv=f'{basePath_csv}/data_science/CSV/zwischenspeicher.csv',allData=[datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f%z'),bundesland,windrad])
            #         breakOutOfMatrix = True
            #         break
            #     else:
            #         print("Wrong input")

            if(breakOutOfMatrix):
                break
            else:
                windrad = windrad+1
        if(breakOutOfMatrix):
            break
        bundesland += 1

label_data(loadProgress,turbine_data, bundesLänder,path_gif,path_jpg,'2022-03-14',['B2','B3','B4'],path_data)
# -----------------------------------------|
# Main Loop for Labeling                   |
#                                          |
# (C) 2022 Felix Nahrstedt, Berlin, Germany|
# email contact@felixnahrstedt.com         |
# -----------------------------------------|
