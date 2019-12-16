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

    
class ee_np_array:
    
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
        self.region = self.geom.bounds()
        
        # Image Collection
        self.image_collection = ee_collection_specifics.ee_collections(self.collection)
 
        # Bands
        self.bands = ee_collection_specifics.ee_bands(self.collection)
    
        # normalized Difference bands and their names
        self.normDiff_bands = ee_collection_specifics.normDiff_bands(self.collection)
        self.normDiff_bands_names = ee_collection_specifics.normDiff_bands_names(self.collection)
        
    def read_datasets(self):
        
        ## Composite
        image = ee_collection_specifics.Composite(self.collection)(self.image_collection, self.startDate, self.stopDate, self.geom)
        
        # Choose the scale
        image =  image.reproject(crs='EPSG:4326', scale=self.scale)
        
        ## Select the bands
        if self.normDiff_bands:
            for n, normDiff_band in enumerate(self.normDiff_bands):
                image_nd = image.normalizedDifference(normDiff_band)
                ## Concatenate images into one multi-band image
                if n == 0:
                    image_nd = image_nd.rename(self.normDiff_bands_names[0])
                    image = ee.Image.cat([image.select(self.bands), image_nd])
                else:
                    image_nd = image_nd.rename(self.normDiff_bands_names[1])
                    image = ee.Image.cat([image, image_nd])
        else:
            image = image.select(self.bands)
            

        # get the lat lon
        latlon = ee.Image.pixelLonLat().addBands([image])
 
        # apply reducer to list
        latlon = latlon.reduceRegion(
          reducer=ee.Reducer.toList(),
          geometry=self.region,
          maxPixels=1e8,
          scale=self.scale);

        # get list of bands
        bands = list(latlon.getInfo().keys())
        bands.remove('latitude'); bands.remove('longitude')
        nbands = len(bands)
    
        # get data into three different arrays
        lats = np.array((ee.Array(latlon.get("latitude")).getInfo()))
        lons = np.array((ee.Array(latlon.get("longitude")).getInfo()))
 
        # get the unique coordinates
        uniqueLats = np.unique(lats)
        uniqueLons = np.unique(lons)
 
        # get number of columns and rows from coordinates
        ncols = len(uniqueLons)    
        nrows = len(uniqueLats)
 
        # determine pixelsizes
        ys = uniqueLats[1] - uniqueLats[0] 
        xs = uniqueLons[1] - uniqueLons[0]
 
        # create an array with dimensions of image
        arr = np.zeros([nrows, ncols, nbands], np.float32) #-9999
 

        for n, band in enumerate(bands):
            data = np.array((ee.Array(latlon.get(band)).getInfo()))

            # fill the array with values
            counter =0
            for y in range(0,len(arr),1):
                for x in range(0,len(arr[0]),1):
                    if lats[counter] == uniqueLats[y] and lons[counter] == uniqueLons[x] and counter < len(lats)-1:
                        counter+=1
                        arr[len(uniqueLats)-1-y,x,n] = data[counter] # we start from lower left corner
        return arr