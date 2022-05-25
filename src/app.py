from datetime import date
import sys
import cv2
from matplotlib import image, pyplot as plt
import numpy as np
import overpy
import io
from PIL import Image
from flask import Flask, redirect, render_template, jsonify, url_for, request
import torch
import torchvision.transforms as transforms
from utils.model.evaluate_single_image import eval_image_with_model
from utils.model.evaluate_single_image import run_inference
from utils.data_preperation.data_information import evaluate_images
from utils.data_labeling.empty_Data import deleteGiffs, deleteImages
from utils.data_collection.convert import convert_img
from utils.data_collection.download_Satellite_tiff import image_preparation
from forms import LatLongForm
from torchvision import models
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import torch.nn as nn

from utils.model.SatelliteTurbinesDataset import Net

path_data = 'Data/'
path_base = ''
path_tiffs = f'{path_data}/Tiffs'
path_jpg = f'{path_data}/Satellite/JPG'
path_gif = f'src/static/gifs'


# -----------------------------------------------------------
# Interface --> Enter Coordinates, Load Image, Make GIF, Predict Image with Neural Net
#
# (C) 2022 Felix Nahrstedt, Berlin, Germany
# email contact@felixnahrstedt.com
# -----------------------------------------------------------

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your secret key'

@app.route('/', methods=('GET', 'POST'))
def index(error = ""):
    form = LatLongForm()
    if form.validate_on_submit():
        return redirect(url_for('imgInput', form=form))
    return render_template('index.html', form=form, error=error)
        
    

@app.route('/imgInput',  methods=['GET', 'POST'])
def imgInput():
    if request.method == 'POST':
        form = LatLongForm(request.form)
        # replace this with an insert into whatever database you're using
        #Fetch Images
        bands = ["B2","B3","B4","B8"]
        key = "fetched"
        #deleteOldSatelliteImages
        deleteImages(f"{path_data}/Satellite")
        deleteGiffs(path_gif)
        datum_img = (date.isoformat(form.datumField.data))
        converter = convert_img(key,datum_img,bands)
        startDate, endDate = image_preparation(path_tiffs,path_jpg,bands,float(form.latitude.data),float(form.longitude.data),datum_img,converter)
        if(startDate==False):
            return render_template('index.html', form=LatLongForm(), error="Could not find usable Images around that Time/Location!")
        _maxMeanBrightness, _maxStdBrightness, imgArr = evaluate_images(path_jpg,key,form.datumField.data,bands)
        newArr = np.array(imgArr)[:,20:40,10:30]
        savePara = list(newArr)
        ax = plt.imshow(cv2.merge([savePara[2],savePara[1],savePara[0]]))
        figure = ax.get_figure()
        path_color = f'static/gifs/color-{str(int(form.latitude.data*1000))}-{str(int(form.longitude.data*1000))}.jpg'
        path_paraNo = f'static/gifs/color-{str(int(form.latitude.data*1000))}-{str(int(form.longitude.data*1000))}-NO.jpg'
        path_paraSo = f'static/gifs/color-{str(int(form.latitude.data*1000))}-{str(int(form.longitude.data*1000))}-SO.jpg'
        path_heatmap = f"static/gifs/heatmap-{str(int(form.latitude.data*1000))}-{str(int(form.longitude.data*1000))}.png"
        path_overlay = f"/static/staticImg/overlayMaxImg-{str(int(form.latitude.data*1000))}-{str(int(form.longitude.data*1000))}.png"
        figure.savefig("src/"+path_color)
        
        parallaxRegion1 = [19]
        parallaxRegion2 = [19]
        parallaxRegion3 = [1]
        parallaxRegion4 = [1]

        newArr = np.repeat(newArr, 3,1)
        newArr = np.repeat(newArr, 3,2)
        print(newArr.shape)
        #if orbit is coming from Northpole
        parallax1 = np.zeros(newArr[1].shape)
        parallax1[1:] = np.delete(newArr[1],parallaxRegion1,0)
        parallax2 = np.empty(np.array(newArr[2]).shape)
        parallax2[1:] = np.delete(newArr[2],parallaxRegion2,0)
        #if orbit is coming from Southpole
        parallax3 = np.zeros(newArr[1].shape)
        parallax3[:-1] = np.delete(newArr[1],parallaxRegion3,0)
        parallax4 = np.empty(np.array(newArr[2]).shape)
        parallax4[:-1] = np.delete(newArr[2],parallaxRegion4,0)

        NoSo = list(np.array([newArr[0],newArr[1],parallax2]))
        SoNo = list(np.array([newArr[0],newArr[1],parallax4]))

        
        cv2.imwrite(path_paraNo,cv2.merge([NoSo[0],NoSo[1],NoSo[2]]))
        cv2.imwrite(path_paraSo,cv2.merge([SoNo[0],SoNo[1],SoNo[2]]))

        
        converter.onlySaveGif(path_gif,imgArr)
        #Get Analysis

        out, _, probs = eval_image_with_model(path_data+"/data_science/Models/Model-pretrained-dense-cropped-20.pt","src/"+path_heatmap,"src"+path_overlay,imgArr)
        
        #Define coordinates of where we want to center our map
        area = [float(form.latitude.data),float(form.longitude.data)]
        
        #Create the map
        my_map = folium.Map(location = area, zoom_start = 13)
        folium.Marker(area, popup = 'Turbine Location').add_to(my_map)
        
        my_map.save('src/templates/map.html')

        inputList = [{"lat": form.latitude.data,"lon":form.longitude.data,"date-Start":startDate,"date-End":endDate, "spin": round(probs[0]*100, 4)}]

        # predicted should be out for next return NOT 1 
        return render_template('imgInput.html',name=f"/static/gifs/{key}-{form.datumField.data}.gif", inputList=inputList, predicted=out,lat=form.latitude.data, lon=form.longitude.data,pathcolor=path_color, heatmap=path_heatmap,overlay = path_overlay)
    form = LatLongForm()
    return render_template('403Error.html')

@app.route('/map')
def map():
    return render_template('map.html')

@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

if __name__ == '__main__':
	app.run(debug=True)