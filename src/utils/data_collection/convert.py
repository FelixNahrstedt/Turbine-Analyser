from matplotlib import image
from osgeo import gdal
import os 
import numpy as np

import imageio
import cv2
# Morphological filtering
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
      print(path_jpg)
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

  def saveToGif(self,path_gif, imgArr,unmatchedArr=None, add=""):
      biggerArray = []
      #biggerArrayUM = []

      for img in imgArr:
          newImg = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)
          biggerArray.append(newImg)
    #   for img in unmatchedArr:
    #       newImg = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)
    #       biggerArrayUM.append(newImg)
      imageio.mimsave(f'{path_gif}/{self.id}-{self.date}{add}.gif', biggerArray)
      #imageio.mimsave(f'{path_gif}/{self.id}-{self.date}-unprepared.gif', biggerArrayUM)

      cv2.imshow('frame',cv2.merge([biggerArray[0],biggerArray[1],biggerArray[2]]))
      cv2.waitKey()
      while(True):
        #This is to check whether to break the first loop
        isclosed=0
        cap = cv2.VideoCapture(f'{path_gif}/{self.id}-{self.date}.gif')
        while (True):

            ret, frame = cap.read()
            # It should only show the frame when the ret is true
            if ret == True:

                cv2.imshow('frame',frame)
                if cv2.waitKey(300) == 27:
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
  
  def onlySaveGif(self,path_gif, imgArr,unmatchedArr=None, add=""):
      biggerArray = []
      #biggerArrayUM = []

      for img in imgArr:
          newImg = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)
          biggerArray.append(newImg)
    #   for img in unmatchedArr:
    #       newImg = cv2.resize(img, (300,300), interpolation = cv2.INTER_AREA)
    #       biggerArrayUM.append(newImg)
      imageio.mimsave(f'{path_gif}/{self.id}-{self.date}{add}.gif', biggerArray)
      
  def saveGrayScaleFromRGB(self,imgPath, imgArr, fileName, bands):
      for i in range(len(imgArr)):
          cv2.imwrite(f'{imgPath}/{fileName}-{bands[i]}.jpg', imgArr[i])

  def image_entropy(img):
        w,h = img.size
        a = np.array(img.convert('RGB')).reshape((w*h,3))
        h,e = np.histogramdd(a, bins=(16,)*3, range=((0,256),)*3)
        prob = h/np.sum(h) # normalize
        prob = prob[prob>0] # remove zeros
        return -np.sum(prob*np.log2(prob))

