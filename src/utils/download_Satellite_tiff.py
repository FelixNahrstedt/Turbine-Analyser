from datetime import date, timedelta
from pathlib import Path
from FireHR.data import *
import collections

from utils.convert import *
collections.Callable = collections.abc.Callable

from PIL import Image
from PIL.TiffTags import TAGS


# -----------------------------------------------------------
# Loading and Preparing Satellite Images 
#
# (C) 2022 Felix Nahrstedt, Berlin, Germany
# email contact@felixnahrstedt.com
# -----------------------------------------------------------

# connection to FireHR
def fetch_satellite_image(path_save, bands, image_Frame,first_Date,last_Date):
    products    = ["COPERNICUS/S2"]  # Product id in google earth engine

    R = RegionST(name         = 'WindTurbine', 
                bbox =  image_Frame,
                scale_meters = 10, 
                time_start   = first_Date, 
                time_end     = last_Date)
    time_window = R.times[0], R.times[-1]

    # Download median composite of the 3 least cloudy images within the time_window
    download_data(R, time_window, products, bands, path_save, use_least_cloudy=1, 
                show_progress=True)

# adapt the input date if no cloudless image is found
def image_preparation(path_tiffs, path_jpg, bands, lat, long, inputDate, converter):
    loop = True
    daySpanStart = 1
    daySpanEnd = 1
    while(loop):
        dateFirst = date.fromisoformat(inputDate)
        dateSec = date.fromisoformat(inputDate)
        dateFirst -= timedelta(days=daySpanStart)
        dateSec += timedelta(days=daySpanEnd)
        left   = long + 0.00165
        right  = long - 0.00185
        bottom = lat + 0.0030
        top    = lat - 0.0005
        try:
            fetch_satellite_image(Path(path_tiffs), bands,[left,bottom,right,top],dateFirst.strftime("%Y/%m/%d"),dateSec.strftime("%Y/%m/%d"))
            loop = False
        except:
            print("error")
            daySpanStart +=5


    converter.saveToJPG(path_jpg,path_tiffs)