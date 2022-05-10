from datetime import date
import sys
import cv2
from matplotlib import image
import numpy as np
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
import torch.nn as nn

from utils.model.SatelliteTurbinesDataset import Net

path_data = 'C:/Users/fe-na/OneDrive/Dokumente/0 - Meine Dateien/Umweltinformatik/Eigene Projekte/Machine Learning/pytorch/sentinel-2-bewegungserkennung/Data'
path_tiffs = f'{path_data}/Tiffs'
path_jpg = f'{path_data}/Satellite/JPG'
path_gif = f'C:/Users/fe-na/OneDrive/Dokumente/0 - Meine Dateien/Umweltinformatik/Eigene Projekte/Machine Learning/pytorch/sentinel-2-bewegungserkennung/src/static/gifs'


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
        bands = ["B2","B3","B4"]
        key = "fetched"
        print(form.datumField.data)
        deleteImages(f"{path_data}/Satellite")
        deleteGiffs(path_gif)
        datum_img = (date.isoformat(form.datumField.data))
        print(datum_img)
        converter = convert_img(key,datum_img,bands)
        startDate, endDate = image_preparation(path_tiffs,path_jpg,bands,float(form.latitude.data),float(form.longitude.data),datum_img,converter)
        if(startDate==False):
            return render_template('index.html', form=LatLongForm(), error="Could not find usable Images around that Time/Location!")
        _maxMeanBrightness, _maxStdBrightness, imgArr = evaluate_images(path_jpg,key,form.datumField.data,bands)
        cv2.imwrite('src/static/staticImg/myImage.png',cv2.merge([imgArr[0],imgArr[1],imgArr[2]]))
        converter.onlySaveGif(path_gif,imgArr)
        #Get Analysis
        inputList = [{"lat": form.latitude.data,"lon":form.longitude.data,"date-Start":startDate,"date-End":endDate}]

        imgArr = []
        for band in bands:
            path = f'{path_jpg}/{key}-{datum_img}-{band}.jpg'
            imgArr.append(image.imread(path))
        out, _ = eval_image_with_model(path_data+"/data_science/Models/Base-Model.pt",imgArr)
        # predicted should be out for next return NOT 1 
        return render_template('imgInput.html',name=f"/static/gifs/{key}-{form.datumField.data}.gif", inputList=inputList, predicted=1)
    form = LatLongForm()
    return render_template('403Error.html')
    
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

#Model

if __name__ == '__main__':
	app.run(debug=True)