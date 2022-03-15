
import imageio
import torch
import numpy as np
import torchvision
import csv


# img_Band2_arr =  imageio.imread('data/jpg_windTurbine/B2.jpg')
# img_Band3_arr =  imageio.imread('data/jpg_windTurbine/B3.jpg')
# img_Band4_arr =  imageio.imread('data/jpg_windTurbine/B4.jpg')

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
    print(f"first thing: {max(meanBrightness)/255}")
    #mean = / amount of pixel and / maximum grayscale val --> 255
    return max(meanBrightness)/255, max(stdBrightness)



def evaluate_images(path_images, id, date, bands):
    img_Band_arr = []
    for band in bands:
        img_Band_arr.append(imageio.imread(f'{path_images}/{id}-{date}-{band}.jpg'))
    maxMeanBright, maxStdBright = maxMeanBrightness(img_Band_arr)
    return maxMeanBright, maxStdBright, img_Band_arr

def appendCsv(pathCsv, id, lat, lon, std, mean):
    with open(pathCsv, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header

        # write the data
        writer.writerow([id,lat,lon,std,mean])

def createCsv(pathCsv, header):
    with open(pathCsv, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

# brightness = maxBrightness(img_Band2_arr,img_Band3_arr,img_Band4_arr)

# img_Band2_Norm = normalizeImages(brightness, img_Band2_arr)
# img_Band3_Norm = normalizeImages(brightness, img_Band3_arr)
# img_Band4_Norm = normalizeImages(brightness, img_Band4_arr)

#imageio.imwrite('data/jpg_windTurbine/B2_Norm.jpg',(img_Band2_Norm))
#imageio.imwrite('data/jpg_windTurbine/B3_Norm.jpg',(img_Band3_Norm))
#imageio.imwrite('data/jpg_windTurbine/B4_Norm.jpg',(img_Band4_Norm))
