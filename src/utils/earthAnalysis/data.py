# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_data.ipynb (unless otherwise specified).

__all__ = ['RegionST', 'extract_region', 'coords2bbox', 'split_region', 'merge_tifs', 'filter_region', 'filter_cloudy',
           'n_least_cloudy', 'download_topography_data', 'download_data', 'download_data_ts', 'get_event_data']

# Cell
import time
import ee
import os
import requests
import rasterio
import pandas as pd
import numpy as np
import zipfile
import json
from IPython.core.debugger import set_trace
from pathlib import Path
import warnings
from fastprogress.fastprogress import progress_bar
from banet.geo import open_tif, merge, Region
from banet.geo import downsample

# Cell
class RegionST(Region):
    "Defines a region in space and time with a name, a bounding box and the pixel size."
    def __init__(self, name:str, bbox:list, pixel_size:float=None, scale_meters:int=None,
                 time_start:str=None, time_end:str=None, time_freq:str='D', time_margin:int=0,
                 shape:tuple=None, epsg=4326):
        if scale_meters is not None and pixel_size is not None:
            raise Exception('Either pixel_size or scale_meters must be set to None.')
        self.name = name
        self.bbox = rasterio.coords.BoundingBox(*bbox) # left, bottom, right, top
        if pixel_size is not None:
            self.pixel_size = pixel_size
        else:
            self.pixel_size = scale_meters/111000
        self.epsg         = epsg
        self.scale_meters = scale_meters
        self._shape       = shape
        self.time_start   = pd.Timestamp(str(time_start))
        self.time_end     = pd.Timestamp(str(time_end))
        self.time_margin  = time_margin
        self.time_freq    = time_freq

    @property
    def shape(self):
        "Shape of the region (height, width)"
        if self._shape is None:
            return (self.height, self.width)
        else: return self._shape
    
    @property
    def times(self):
        "Property that computes the date_range for the region."
        tstart = self.time_start - pd.Timedelta(days=self.time_margin)
        tend = self.time_end + pd.Timedelta(days=self.time_margin)
        return pd.date_range(tstart, tend, freq=self.time_freq)

    @classmethod
    def load(cls, file, time_start=None, time_end=None):
        "Loads region information from json file"
        with open(file, 'r') as f:
            args = json.load(f)
        if time_start is None:
            time_start = args['time_start']
        if time_end is None:
            time_end = args['time_end']
        return cls(args['name'], args['bbox'], args['pixel_size'],
                   time_start=time_start, time_end=time_end)

    @classmethod
    def getInfo(self):
        a = {self.epsg, self.scale_meters, self._shape, self.time_start ,self.time_end,self.time_margin,self.time_freq}
        print(a)

def extract_region(df_row, cls=Region):
    "Create Region object from a row of the metadata dataframe."
    if issubclass(cls, RegionST):
        return cls(df_row.event_id, df_row.bbox, df_row.pixel_size,
                   df_row.time_start, df_row.time_end)
    elif issubclass(cls, Region):
        return cls(df_row.event_id, df_row.bbox, df_row.pixel_size)
    else: raise NotImplemented('cls must be one of the following [Region, RegionST]')

# Cell
def coords2bbox(lon, lat, pixel_size):
    return [lon.min(), lat.min(), lon.max()+pixel_size, lat.max()+pixel_size]

def split_region(region:RegionST, size:int, cls=Region):
    lon, lat = region.coords()
    Nlon = (len(lon)//size)*size
    Nlat = (len(lat)//size)*size
    lons = [*lon[:Nlon].reshape(-1, size), lon[Nlon:][None]]
    lats = [*lat[:Nlat].reshape(-1, size), lat[Nlat:][None]]
    if len(lats[-1].reshape(-1)) == 0 and len(lons[-1].reshape(-1)) == 0:
        lons = lons[:-1]
        lats = lats[:-1]
    #lons = lon.reshape(-1, size)
    #lats = lat.reshape(-1, size)
    if issubclass(cls, RegionST):
        return [cls('', coords2bbox(ilon, ilat, region.pixel_size),
                    pixel_size=region.pixel_size, time_start=region.time_start,
                    time_end=region.time_end, time_freq=region.time_freq,
                    time_margin=region.time_margin) for ilon in lons for ilat in lats]
    elif issubclass(cls, Region):
        return [cls('', coords2bbox(ilon, ilat, region.pixel_size), pixel_size=region.pixel_size)
            for ilon in lons for ilat in lats]
    else: raise NotImplemented('cls must be one of the following [Region, RegionST]')

    return

def merge_tifs(files:list, fname:str, delete=False):
    data, tfm = merge([open_tif(str(f)) for f in files])
    data = data.squeeze()
    fname = Path(files[0]).parent/fname
    profile = open_tif(str(files[0])).profile
    with rasterio.Env():
        height, width = data.shape
        profile.update(width=width, height=height, transform=tfm, compress='lzw')
        with rasterio.open(str(fname), 'w', **profile) as dst:
            dst.write(data, 1)
    if delete:
        for f in files: os.remove(f)

# Cell
def filter_region(image_collection:ee.ImageCollection, region:RegionST, times:tuple, bands=None):
    image_collection = image_collection.filterDate(times[0], times[1])
    geometry = ee.Geometry.Rectangle(region.bbox)
    image_collection = image_collection.filterBounds(geometry)
    if bands is not None:
        image_collection = image_collection.select(bands)
    return image_collection

def filter_cloudy(image_collection:ee.ImageCollection, max_cloud_fraction=0.2):
    return image_collection.filterMetadata(
        'CLOUDY_PIXEL_PERCENTAGE', 'not_greater_than', max_cloud_fraction)


def download_data(R:RegionST, times, products, bands, path_save, scale=None, max_cloud_fraction=None, download_crop_size=1000, show_progress=False):
    if scale is None: scale = R.scale_meters
    ee.Initialize()
    path_save.mkdir(exist_ok=True, parents=True)
    if not ((path_save/f'download.{bands[0]}.tif').is_file() and
           (path_save/f'download.{bands[1]}.tif').is_file() and
           (path_save/f'download.{bands[2]}.tif').is_file()):
        sR = [R] if min(R.shape) <= download_crop_size else split_region(R, size=download_crop_size, cls=RegionST)
        fsaves = []
        #for j, R in tqdm(enumerate(sR), total=len(sR)):
        loop = enumerate(sR) if not show_progress else progress_bar(enumerate(sR),total=len(sR))
        for j, R in loop:
            region = (f"[[{R.bbox.left}, {R.bbox.bottom}], [{R.bbox.right}, {R.bbox.bottom}], " +
                       f"[{R.bbox.right}, {R.bbox.top}], [{R.bbox.left}, {R.bbox.top}]]")

            if not ((path_save/f'download.{bands[0]}_{j}.tif').is_file() and
                   (path_save/f'download.{bands[1]}_{j}.tif').is_file() and
                   (path_save/f'download.{bands[2]}_{j}.tif').is_file()):
                # Merge products to single image collection
                imCol = ee.ImageCollection(products[0])
                
                for i in range(1, len(products)):
                    imCol = imCol.merge(ee.ImageCollection(products[i]))
                imCol = filter_region(imCol, R, times=times, bands=bands)
                if max_cloud_fraction is not None:
                    imCol = filter_cloudy(imCol, max_cloud_fraction=max_cloud_fraction)
                im = imCol.median()
                imCol = ee.ImageCollection([im])
                colList = imCol.toList(imCol.size())
                
                for i in range(colList.size().getInfo()):
                    image = ee.Image(colList.get(i))
                    
                    fname = 'download'
                    #fname = image.get('system:id').getInfo().split('/')[-1]
                    fnames_full = [f'{fname}.{b}.tif' for b in bands]
                    fnames_partial0 = [f'{fname}.{b}_{j}.tif' for b in bands]
                    fnames_full = all([(path_save/f).is_file() for f in fnames_full])
                    fnames_partial = all([(path_save/f).is_file() for f in fnames_partial0])
                    if not fnames_full:
                        fsaves.append([path_save/f for f in fnames_partial0])
                        if not fnames_partial:
                            zip_error = True
                            for i in range(10): # Try 10 times
                                if zip_error:
                                    try:

                                        url = image.getDownloadURL(
                                            {"scale": scale, "crs": 'EPSG:4326',
                                             "region": f'{region}'})
                                        r = requests.get(url)
                                        with open(str(path_save/'data.zip'), 'wb') as f:
                                            f.write(r.content)
                                        
                                        with zipfile.ZipFile(str(path_save/'data.zip'), 'r') as f:
                                            files = f.namelist()
                                            f.extractall(str(path_save))
                                        os.remove(str(path_save/'data.zip'))
                                        zip_error = False
                                    except Exception as e:
                                        zip_error = True
                                        os.remove(str(path_save/'data.zip'))
                                        time.sleep(10)
                            if zip_error: raise Exception(f'Failed to process {url}')
                            for f in files:
                                f = path_save/f
                                os.rename(str(f), str(path_save/f'{f.stem}_{j}{f.suffix}'))
        # Merge files
        suffix = '.tif'

        files = path_save.ls(include=[suffix])
        #files = np.unique(fsaves)
        files = [o.stem for o in files]
        ref = np.unique(['_'.join(o.split('_')[:-1])
                         for o in files if len(o.split('_')[-1]) < 6])
        ids = np.unique([int(o.split('_')[-1])
                         for o in files if len(o.split('_')[-1]) < 6])
        #file_groups = [[path_save/f'{r}_{i}{suffix}' for i in ids] for r in ref]
        file_groups = [[path_save/f'{r}_{i}{suffix}' for i in ids
                    if f'{r}_{i}' in files] for r in ref]
        for fs in file_groups:
            if len(fs) < 500:
                fsave = '_'.join(fs[0].stem.split('_')[:-1]) + suffix
                merge_tifs(fs, fsave, delete=True)
            else:
                fs_break = np.array(fs)[:(len(fs)//500)*500].reshape(len(fs)//500,-1).tolist()
                if len(fs[(len(fs)//500)*500:]) > 0:
                    fs_break.append(fs[(len(fs)//500)*500:])
                for fsi, fs2 in enumerate(fs_break):
                    fsave = '_'.join(fs2[0].stem.split('_')[:-1]) + f'_break{fsi}' + suffix
                    merge_tifs(fs2, fsave, delete=True)

        files = path_save.ls(include=[suffix, '_break'])
        files = [o.stem for o in files]
        ref = np.unique(['_'.join(o.split('_')[:-1])
                         for o in files if len(o.split('_')[-1]) < 11])
        ids = np.unique([o.split('_')[-1]
                         for o in files if len(o.split('_')[-1]) < 11])
        #file_groups = [[path_save/f'{r}_{i}{suffix}' for i in ids] for r in ref]
        file_groups = [[path_save/f'{r}_{i}{suffix}' for i in ids
                    if f'{r}_{i}' in files] for r in ref]
        for fs in file_groups:
            fsave = '_'.join(fs[0].stem.split('_')[:-1]) + suffix
            merge_tifs(fs, fsave, delete=True)
