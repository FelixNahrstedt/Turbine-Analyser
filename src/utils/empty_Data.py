import os

# -----------------------------------------------------------
# deletes tiffs and unsused material from hard drive
#
# (C) 2022 Felix Nahrstedt, Berlin, Germany
# email contact@felixnahrstedt.com
# -----------------------------------------------------------

def deleteImages(filePath):
  #remove forlders
  import os
  for root, dirs, files in os.walk(filePath, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

  jpg_windTurbine = filePath + "/JPG"
  gif_windTurbine = filePath + "/GIF"
  #tif_windTurbine = filePath + "/locations"
  tif_windTurbine = filePath + "/Tiffs"

  os.mkdir(jpg_windTurbine)
  os.mkdir(tif_windTurbine)
  os.mkdir(gif_windTurbine)

def deleteGiffs(filePath):
  #remove forlders
  import os
  for root, dirs, files in os.walk(filePath, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
