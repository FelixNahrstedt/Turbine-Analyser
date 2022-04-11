
import random
import imageio
from matplotlib import spines
import pandas as pd
import torch
import numpy as np
import torchvision
from PIL import Image
import csv
import matplotlib.pyplot as plt
import mpl_toolkits
import matplotlib.pyplot as plt
from utils.convert import convert_img
import pandas as pd
from numpy.random import RandomState
from scipy.ndimage import gaussian_filter
import utils.img_plots 
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------
# all functions that evaluate the image files
#
# (C) 2022 Felix Nahrstedt, Berlin, Germany
# email contact@felixnahrstedt.com
# -----------------------------------------------------------

def calcBrightness(img_arr):
    sum = 0
    for pixelArr in img_arr:
      for pixel in pixelArr:
         sum = sum+pixel
    return sum

def normalizeImages(minBrightness, image):
    faktor = minBrightness/calcBrightness(image)
    newImage = (faktor*image).astype(np.uint8)
    return newImage

def maxMeanBrightness(args):
    meanBrightness = []
    stdBrightness = []
    for element in args:
        meanBrightness.append(element.mean(axis=0).mean())
        stdBrightness.append(element.std(axis=0).std())
    #mean = / amount of pixel and / maximum grayscale val --> 255
    return max(meanBrightness)/255, max(stdBrightness)

def evaluate_images(path_images, id, date, bands):
    img_Band_arr = []
    for band in bands:
        img_Band_arr.append(imageio.imread(f'{path_images}/{id}-{date}-{band}.jpg'))
    maxMeanBright, maxStdBright = maxMeanBrightness(img_Band_arr)
    return maxMeanBright, maxStdBright, img_Band_arr
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
def display_data(path_data):
    file = open(path_data)
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    # Define Data
    #0 = Drehendes Windrad, 
    goodQual = []
    goodQual2 = []

    medQual1 = []
    medQual2 = []

    badQual1 = []
    badQual2 = []
    print(header)
    amountSpin = 0
    amountNotSpin = 0
    amountUndetected = 0
    goodVisibleSpin = 0
    print(f'Insgesamt: {len(rows)}')
    for row in rows:
        if(row[3] =='0'):
            amountSpin = amountSpin+1
        if(row[3] =='1'):
            amountNotSpin = amountNotSpin +1
        if(row[3] =='2'):
            amountUndetected = amountUndetected +1
        if(row[3] == '0' and (row[4] == '1' or row[4] == '2' )):
            goodVisibleSpin = goodVisibleSpin +1
        if(row[4] =='1' or row[4] =='2'):
            goodQual.append(row[5])
            goodQual2.append(row[6])

        elif(row[4] =='3' or row[4] == '4'):
            medQual1.append(row[5])
            medQual2.append(row[6])
        else:
            badQual1.append(row[5])
            badQual2.append(row[6])
    #1 = Stehendes Windrad, 
    #2 = Nicht erkennbar
    print(f'Of these Spinning: {amountSpin}, Not Spinning: {amountNotSpin}, undetectable: {amountUndetected}')
    print(f'spinning and in class 1-2: {goodVisibleSpin}')
    # Create Figure


    fig = plt.figure(figsize = (8,6))
    
    # Create Plot

    plt.scatter(np.asarray(goodQual).astype(np.float16), NormalizeData(np.asarray(goodQual2).astype(np.float16)), s=10, color='green')
    plt.scatter(np.asarray(medQual1).astype(np.float16),  NormalizeData(np.asarray(medQual2).astype(np.float16)),  s=3, color='blue')
    plt.scatter(np.asarray(badQual1).astype(np.float16),  NormalizeData(np.asarray(badQual2).astype(np.float16)), s=3, color='red')

    # Show plot

    plt.show()

#! MOVE TO load_data.py !! to-do
# -----------------------------------------------------------
# all functions for saving/loading data
#
# (C) 2022 Felix Nahrstedt, Berlin, Germany
# email contact@felixnahrstedt.com
# -----------------------------------------------------------

def appendCsv(pathCsv, id, lat, lon, std, mean, label, date, qual, region):
    with open(pathCsv, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header

        # write the data
        writer.writerow([id,lat,lon,label,qual,mean,std,date,region])

def appendCsv_open(pathCsv, allData):
    with open(pathCsv, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
         # write the data
        writer.writerow(allData)

def createCsv(pathCsv, header):
    with open(pathCsv, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

def unspin_turbines(path_data, path_jpg, path_gif,path_unspinned):
    file = open(path_data)
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    print(header)
    goodQualSpin =  []
    for row in rows:
         if(row[3] == '0' and (row[4] == '1' or row[4] == '2' )):
            goodQualSpin.append([row[0],row[7]])
    
    for satImg in goodQualSpin:

        bands = ["B2","B3","B4"]
        imgArr = []
        originalFile = f'{satImg[0]}-{satImg[1]}'
        fakeFile = f'{satImg[0]}-unspin-{satImg[1]}'
        for band in bands: 
            imgArr.append(np.asarray(Image.open(f'{path_jpg}/{originalFile}-{band}.jpg')))


        id = "280498836"
        bands = ["B2","B3","B4"]
        date = "2022-03-14"
        newImgPlot = utils.img_plots.img_plots(path_jpg,id,bands, date)
        def saltPepperChannel(frame,size):
                for pixelRow in range(size-1):
                    for pixel in range(size-1): 
                        newVal = frame[pixelRow,pixel]
                        movementInt = random.randint(0,10)

                        chooseRand = random.randint(0,80)

                        #every second pixel either gets down or upscaled in order to add Noise
                        if(chooseRand >0 and chooseRand < 20):
                            newVal = newVal - movementInt
                        elif(chooseRand >20 and chooseRand < 40):
                            newVal = newVal + movementInt

                        if(newVal>=253):
                            newVal = 255

                        if(newVal <= 2):
                            newVal = 1

                        frame[pixelRow,pixel] = newVal 

                
                return frame
                


        transformed = newImgPlot.histogram_Transform(np.asarray(imgArr))
        imgArr = []
        imgArr.append(saltPepperChannel(transformed[0],40))
        imgArr.append(saltPepperChannel(transformed[1],40))
        imgArr.append(saltPepperChannel(transformed[2],40))
        convertTransform = convert_img(id,date,imgArr)
        #convertTransform.saveToGif(path_jpg,imgArr)
        convertTransform.saveGrayScaleFromRGB(path_unspinned,imgArr,fakeFile,bands)

def createSubCsv(turbine_data, csvData):
    subsetSize = 1064
    header = ['id', 'latitude','longitude','label','quality','max_mean-bright', 'max_std_bright', 'date','region']
    not_spinning = csvData+"not_spinning.csv"
    spinning = csvData+"spinning.csv"
    undetected = csvData+"undetected.csv"

    createCsv(not_spinning, header)
    createCsv(spinning, header)
    createCsv(undetected, header)

    file = open(turbine_data)
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    print(len(rows))
    spines = 0
    nospins = 0
    undetectedes =0
    print(header)
    for row in rows:
        if(row[3] == '0' and (row[4] == '1' or row[4] == '2' or row[4] == '3')):
            if(spines < subsetSize):
                spines = spines +1
                #append date and ID for later image loading
                appendCsv_open(spinning,row)
            if(row[4] == '1' or row[4] == '2'):
                if(nospins<subsetSize):
                    nospins = nospins+1
                    row[0] = row[0]+"-unspin"
                    row[3] = 1
                    appendCsv_open(not_spinning,row)

        if(row[3] == '2'):
            if(undetectedes<subsetSize):
                undetectedes = undetectedes +1
                appendCsv_open(undetected,row)
        if(row[3] == '1'):
            if(nospins<1064):
                nospins = nospins +1
                appendCsv_open(not_spinning,row)
    
    print(spines, nospins, undetectedes)

def splitTrainTest(spinningCsv, notSpinningCsv,UndetectedCsv, TrainCsv, TestCsv):
    csvreader = csv.reader(spinningCsv)

    dfSpinTrain,dfSpinTest = csvToTrainTest(spinningCsv)
    dfNoSpinTrain,dfNoSpinTest = csvToTrainTest(notSpinningCsv)
    dfUndetectedTrain,dfUndetectedTest = csvToTrainTest(UndetectedCsv)
    
    dfTrain = pd.concat([dfSpinTrain,dfNoSpinTrain,dfUndetectedTrain])
    dfTest = pd.concat([dfSpinTest,dfNoSpinTest,dfUndetectedTest])
    print(f"Train: {len(dfTrain)}")
    print(f"Train: {len(dfTest)}")

    #shuffle
    dfTrainShuffled = dfTrain.sample(frac=1)
    dfTestShuffled = dfTest.sample(frac=1)
    dfTrainShuffled.to_csv(TrainCsv, index=False)
    dfTestShuffled.to_csv(TestCsv, index=False)


def csvToTrainTest(filePath):
    data = pd.read_csv(filePath)
    data['split'] = np.random.randn(data.shape[0], 1)

    rng = RandomState()

    train = data.sample(frac=0.8, random_state=rng)
    test = data.loc[~data.index.isin(train.index)]  
    print(f"Train: {len(train)}")
    print(f"Test: {len(test)}")
    return train, test
    