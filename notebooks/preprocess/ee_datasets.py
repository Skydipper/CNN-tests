from urllib.request import urlopen
import zipfile
import gzip
import rasterio
import os, urllib
import sys
import shutil
import numpy as np
import math
import ee_collection_specifics
import ee

    
def download_image_tif(image, download_zip, scale, region = None):
    
    Vizparam = {'scale': scale, 'crs': 'EPSG:4326'}
    if region:
        Vizparam['region'] = region
    
   
    url = image.getDownloadUrl(Vizparam)     

    data = urlopen(url)
    with open(download_zip, 'wb') as fp:
        while True:
            chunk = data.read(16 * 1024)
            if not chunk: break
            fp.write(chunk)
            
    # extract the zip file transformation data
    z = zipfile.ZipFile(download_zip, 'r')
    target_folder_name = download_zip.split('.zip')[0]
    z.extractall(target_folder_name)
 
        
def load_tif_bands(path, files):
    data = np.array([]) 
    for n, file in enumerate(files):
        image_path = path+file
        image = rasterio.open(image_path)
        data = np.append(data, image.read(1))
    data = data.reshape((n+1, image.read(1).shape[0], image.read(1).shape[1]))
    data = np.moveaxis(data, 0, 2)
    
    return data


class ee_datasets:
    
    def __init__(self, point, buffer, startDate, stopDate, scale, collection):
        """
        Class used to get the datasets from Earth Engine
        Parameters
        ----------
        point : list
            A list of two [x,y] coordinates with the center of the area of interest.
        buffer : number
            Buffer in meters
        startDate : string
        stopDate : string
        scale: number
            Pixel size in meters.
        collection: string
            Name of each collection.

        """
        
        self.point = point
        self.buffer = buffer
        self.startDate = startDate
        self.stopDate = stopDate       
        self.scale = scale 
        self.collection = collection
        
        # Area of Interest
        self.geom = ee.Geometry.Point(self.point).buffer(self.buffer)
        self.region = self.geom.bounds().getInfo()['coordinates']
        
        # Image Collection
        self.image_collection = ee_collection_specifics.ee_collections(self.collection)
 
        # Bands
        self.bands = ee_collection_specifics.ee_bands(self.collection)
    
        # normalized Difference bands
        self.normDiff_bands = ee_collection_specifics.normDiff_bands(self.collection)
        
    def read_datasets(self):
        
        ## Composite
        image = ee_collection_specifics.Composite(self.collection)(self.image_collection, self.startDate, self.stopDate, self.geom)
        
        ## Calculate normalized Difference
        if self.normDiff_bands:
            for n, normDiff_band in enumerate(self.normDiff_bands):
                image_nd = image.normalizedDifference(normDiff_band)
                ## Concatenate images into one multi-band image
                if n == 0:
                    image = ee.Image.cat([image.select(self.bands), image_nd])
                else:
                    image = ee.Image.cat([image, image_nd])
        else:
            image = image.select(self.bands)
        
        # Choose the scale
        image =  image.reproject(crs='EPSG:4326', scale=self.scale)
            
        # Download images as tif
        download_image_tif(image, 'data.zip', scale = self.scale, region = self.region)
        
        # Load data
        directory = "./data/"

        files = sorted(f for f in os.listdir(directory) if f.endswith('.' + 'tif'))
        
        data = load_tif_bands(directory, files)
        
        # Remove data folders and files
        file="data.zip"
        ## If file exists, delete it ##
        if os.path.isfile(file):
            os.remove(file)
        else:    ## Show an error ##
            print("Error: %s file not found" % file)
        ## Try to remove tree; if failed show an error using try...except on screen
        folder = "./data"
        try:
            shutil.rmtree(folder)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))
            
        return data