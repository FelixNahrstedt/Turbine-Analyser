#geo locations
from utils.data_preperation.data_information import appendCsv_open
from utils.data_labeling.empty_Data import deleteImages
from utils.data_collection.download_Satellite_tiff import image_preparation
from utils.model.evaluate_single_image import eval_image_with_model
from utils.data_collection.convert import convert_img
from utils.data_preperation.data_information import evaluate_images
from utils.data_labeling.load_data import import_json


def useOnCountries(name, date, imgPath,basePath, tiff_path, gif_path):
    bands = ["B2","B3","B4"] 
    jsonData, turbine_keys = import_json(name)
    #Sorting Loop
    windrad = 0
    indices = []
    while(windrad<len(turbine_keys)):
        key = turbine_keys[windrad]
        windrad +=1
        data =  jsonData[key]
    # create large gif and show it 
        
        print('\n')
        print(data['latitude'],data['longitude'])
        print(f'Windräder durch = {windrad}')
        deleteImages(basePath)
        converter = convert_img(key,date,bands)
        startDate, endDate = image_preparation(tiff_path,imgPath, bands,data['latitude'],data['longitude'],date,converter)
        if(startDate == False):
            continue
        maxMeanBrightness, maxStdBrightness, imgArr = evaluate_images(imgPath,key,date,bands)
        #matchingImgs = img_plots(imgPath,key,bands,date)
        label, out = eval_image_with_model("Data/data_science/Models/Sentinel_2_pretrained-Dense.pt",imgArr)
        print(label)
        appendCsv_open("Data/data_science/CSV/raw-Data/Spain.csv",[key, data['latitude'],data['longitude'],out, date])
        if(label=="spinning"):
            converter.saveToGif(gif_path,imgArr)
            indices.append(windrad-1)
        print(len(indices))

      
        

#createCsv(f'Data/data_science/CSV/raw-Data/Spain.csv',['id', 'latitude','longitude','label', 'date'])
useOnCountries("Australia","2021-12-13","Data/data_science/img_database_Australia","Data/Satellite","Data/Tiffs","Data/Satellite/GIF")