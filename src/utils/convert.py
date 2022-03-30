from cv2 import waitKey
from matplotlib import image
from osgeo import gdal
import os 
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as rxr
import imageio
from PIL import Image, ImageEnhance,ImageChops
import cv2
# Morphological filtering
from skimage.morphology import opening
from skimage.morphology import disk
from skimage import exposure
from skimage import filters
import json
import argparse
import earthpy as et
import math
import earthpy.spatial as es
import earthpy.plot as ep

os.environ['GDAL_PAM_ENABLED'] = 'NO'

# -----------------------------------------------------------
# convert_img is supposed to convert files into different image formats
#
# (C) 2022 Felix Nahrstedt, Berlin, Germany
# email contact@felixnahrstedt.com
# -----------------------------------------------------------
class convert_img:
  id,date,bands = -1,"",[]
  add = ""

  def __init__(self, id, date, bands,add=""):
      self.id = id
      self.date = date
      self.bands = bands
      self.add = add

  def saveToJPG(self,path_jpg, path_tif):    
      options_list = [
          '-ot Byte',
          '-of JPEG',
          '-b 1',
          '-scale',
      ]           
      
      options_string = " ".join(options_list)

      for band in self.bands:
        gdal.Translate(
            f'{path_jpg}/{self.id}-{self.date}-{band}.jpg',
            f'{path_tif}/download.{band}{self.add}.tif',
            options=options_string,
        )
      
      #load JPG

      #deleteOldJPG

      #StretchJPG
    
  def image_stacked(self, path):
      imgArr = []
      for band in self.bands:
          path = f'{path}/{self.id}-{self.date}-{band}.jpg'
          imgArr.append(image.imread(path))
      stacked = (np.dstack((imgArr[0],imgArr[1],imgArr[2]))).astype(np.uint8)
      return stacked 

  def saveToGif(self,path_gif, imgArr, add=""):
      biggerArray = []
      for img in imgArr:
          newImg = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)
          biggerArray.append(newImg)
      np.shape(biggerArray)
      imageio.mimsave(f'{path_gif}/{self.id}-{self.date}{add}.gif', biggerArray)

      cv2.imshow('frame',cv2.merge([biggerArray[0],biggerArray[1],biggerArray[2]]))
      cv2.waitKey()
      while(True):
        #This is to check whether to break the first loop
        isclosed=0
        cap = cv2.VideoCapture(f'{path_gif}/{self.id}-{self.date}{add}.gif')
        while (True):

            ret, frame = cap.read()
            # It should only show the frame when the ret is true
            if ret == True:

                cv2.imshow('frame',frame)
                if cv2.waitKey(50) == 27:
                    # When esc is pressed isclosed is 1
                    isclosed=1
                    break
            else:
                break
        # To break the loop if it is closed manually
        if isclosed:
            break

        # When everything done, release
        # the video capture object
      cap.release()

      # Closes all the frames
      cv2.destroyAllWindows()

    

path_data = 'C:/Users/fe-na/OneDrive/Dokumente/0 - Meine Dateien/Umweltinformatik/Eigene Projekte/Machine Learning/pytorch/sentinel-2-bewegungserkennung/Data'
path_jpg = f'{path_data}/data_science/images_while_looping'

def image_entropy(img):
    w,h = img.size
    a = np.array(img.convert('RGB')).reshape((w*h,3))
    h,e = np.histogramdd(a, bins=(16,)*3, range=((0,256),)*3)
    prob = h/np.sum(h) # normalize
    prob = prob[prob>0] # remove zeros
    return -np.sum(prob*np.log2(prob))

b2 = "280498836-2022-03-14-B2.jpg"
b3 = "280498836-2022-03-14-B3.jpg"
b4 = "280498836-2022-03-14-B4.jpg"

frame1 = np.asarray(Image.open(f'{path_jpg}/{b2}'))
frame2 = np.asarray(Image.open(f'{path_jpg}/{b3}'))
frame3 = np.asarray(Image.open(f'{path_jpg}/{b4}'))
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

flow = None
flow = cv2.calcOpticalFlowFarneback(prev=frame1,
                                    next=frame2, flow=flow,
                                    pyr_scale=0.2, levels=5, winsize=3,
                                    iterations=10, poly_n=3, poly_sigma=0,
                                    flags=10)
mag1, ang1 = cv2.cartToPolar(flow[...,0], flow[...,1])
img1 =cv2.normalize(mag1,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

flow = cv2.calcOpticalFlowFarneback(prev=frame2,
                                    next=frame3, flow=flow,
                                    pyr_scale=0.8, levels=15, winsize=5,
                                    iterations=10, poly_n=5, poly_sigma=0,
                                    flags=10)
mag2, ang2 = cv2.cartToPolar(flow[...,0], flow[...,1])
img2 =cv2.normalize(mag2,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

flow = cv2.calcOpticalFlowFarneback(prev=frame3,
                                    next=frame1, flow=flow,
                                    pyr_scale=0.8, levels=15, winsize=5,
                                    iterations=10, poly_n=5, poly_sigma=0,
                                    flags=10)
mag3, ang3 = cv2.cartToPolar(flow[...,0], flow[...,1])
img3 =cv2.normalize(mag3,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
# Change here

imgStack = [frame1,frame1,frame1,frame2,frame2,frame2,frame3,frame3,frame3]
imageio.mimsave(f'{path_jpg}/flowMotion.gif', [img1,img1,img2,img2,img3,img3])
imageio.mimsave(f'{path_jpg}/turbineSpin.gif', imgStack)

