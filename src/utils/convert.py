from osgeo import gdal
import os 
import imageio
os.environ['GDAL_PAM_ENABLED'] = 'NO'

def saveToJPG(bands,path_jpg, path_tif, img_id,date):    
  options_list = [
      '-ot Byte',
      '-of JPEG',
      '-b 1',
      '-scale'
  ]           
  
  options_string = " ".join(options_list)
  for band in bands:
    gdal.Translate(
        f'{path_jpg}/{img_id}-{date}-{band}.jpg',
        f'{path_tif}/download.{band}.tif',
        options=options_string,
    )

def saveToGif(path_gif,img_id,date, imgArr):
  imageio.mimsave(f'{path_gif}/{img_id}-{date}.gif', imgArr)