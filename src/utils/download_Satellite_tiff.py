from datetime import date, timedelta
from pathlib import Path
from FireHR.data import *
import collections

from utils.convert import saveToJPG
collections.Callable = collections.abc.Callable
# Bounding box coordinates

def fetch_satellite_image(path_save, bands, image_Frame,first_Date,last_Date):
    products    = ["COPERNICUS/S2"]  # Product id in google earth engine

    R = RegionST(name         = 'TeslaGigaBerlin', 
                bbox =  image_Frame,
                scale_meters = 10, 
                time_start   = first_Date, 
                time_end     = last_Date)

    # Download time series
    # download_data_ts(R, products, bands, path_save)

    time_window = R.times[0], R.times[-1]

    # Download median composite of the 3 least cloudy images within the time_window
    download_data(R, time_window, products, bands, path_save, use_least_cloudy=3, 
                show_progress=True)

def image_preparation(path_tiffs, path_jpg, bands, lat, long, inputDate, id):
    dateFirst = date.fromisoformat(inputDate)
    dateSec = date.fromisoformat(inputDate)
    dateFirst -= timedelta(days=1)
    dateSec += timedelta(days=1)
    left   = long + 0.00175
    right  = long - 0.00175
    bottom = lat + 0.0025
    top    = lat - 0.0005
    fetch_satellite_image(Path(path_tiffs), bands,[left,bottom,right,top],dateFirst.strftime("%Y/%m/%d"),dateSec.strftime("%Y/%m/%d"))
    saveToJPG(bands,path_jpg,path_tiffs, id, inputDate)