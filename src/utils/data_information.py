
import imageio
import torch
import numpy as np
import torchvision
from PIL import Image
import csv
import matplotlib.pyplot as plt
import os



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

