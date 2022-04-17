from datetime import date
import sys
from matplotlib import image
import numpy as np
from flask import Flask, redirect, render_template, jsonify, url_for, request
import torch
import torchvision.transforms as transforms
from utils.data_preperation.data_information import evaluate_images
from utils.data_labeling.empty_Data import deleteGiffs, deleteImages
from utils.data_collection.convert import convert_img
from utils.data_collection.download_Satellite_tiff import image_preparation
from forms import LatLongForm


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
def index():
    form = LatLongForm()
    if form.validate_on_submit():
        return redirect(url_for('imgInput', form=form))
    return render_template('index.html', form=form)


@app.route('/validate',  methods=['GET', 'POST'])
def validate():
    if request.method == 'POST':
        form = LatLongForm(request.form)
        
    

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
        _maxMeanBrightness, _maxStdBrightness, imgArr = evaluate_images(path_jpg,key,form.datumField.data,bands)
        converter.onlySaveGif(path_gif,imgArr)
        #Get Analysis
        inputList = [{"lat": form.latitude.data,"lon":form.longitude.data,"date-Start":startDate,"date-End":endDate}]

        imgArr = []
        for band in bands:
            path = f'{path_jpg}/{key}-{datum_img}-{band}.jpg'
            imgArr.append(image.imread(path))
        transform = transforms.Compose([transforms.ToPILImage(),
                        transforms.Resize((40, 40)),
                        transforms.ToTensor()],
                    )
        stacked = (np.dstack((imgArr[0],imgArr[1],imgArr[2]))).astype(np.uint8)
        item=transform(stacked)
        out = run_inference(item)

        return render_template('imgInput.html',name=f"/static/gifs/{key}-{form.datumField.data}.gif", inputList=inputList, predicted=out)
    form = LatLongForm()
    return render_template('403Error.html')
    

#Model
model = Net()
classes = ["spinning","not spinning", "undetected"]
model.load_state_dict(torch.load(path_data+"/data_science/Models/Sentinel_2.pt"))
model.eval()
def run_inference(in_tensor):
    with torch.no_grad():
        # LunaModel takes a batch and outputs a tuple (scores, probs)
        out_tensor = model(in_tensor.unsqueeze(0)).squeeze(0)
        probs = out_tensor.tolist()
        print(f'Spin: {probs[0]}, No-Spin: {probs[1]}, Undetected: {probs[2]}' )
        out = probs.index(max(probs))
        print(classes[out])
        return classes[out]


@app.route("/predict", methods =["POST"])
def predict(bands = ["B2","B3","B4"]):
    imgArr = []
    for band in bands:
        path = f'{path_jpg}/{"8043435502"}-{"2022-03-14"}-{band}.jpg'
        imgArr.append(image.imread(path))
    transform = transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize((40, 40)),
                    transforms.ToTensor()],
                )
    stacked = (np.dstack((imgArr[0],imgArr[1],imgArr[2]))).astype(np.uint8)
    item=transform(stacked)
    out = run_inference(item)
    return jsonify(out)

if __name__ == '__main__':
	app.run(debug=True)