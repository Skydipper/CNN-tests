from urllib.request import urlopen
import zipfile
import gzip
import rasterio
import os, urllib
import sys
import shutil
import numpy as np
import math
from preprocess import ee_collection_specifics_TFRecord
import ee
import h5py


class ee_dataset:
    
    def __init__(self, points, buffer, startDate, stopDate, scale, patch_size, file_name, collections):
        """
        Class used to get the datasets from Earth Engine
        Parameters
        ----------
        points : list
            A list of two [x,y] coordinates lists with the centers of the area of interest.
        buffer : number
            Buffer in meters
        startDate : string
        stopDate : string
        scale: number
            Pixel size in meters.
        patch_size: list
            list with the [height,width] ot the patch.
        file_name: list of string
            File names prefix.
        collections: list of string
            Name of each collection.
        """
        
        self.points = points
        self.buffer = buffer
        self.startDate = startDate
        self.stopDate = stopDate       
        self.scale = scale 
        self.file_name = file_name
        self.collections = collections
        self.patch_size = patch_size
        
        
        # Google Cloud Bucket
        self.bucket = 'skydipper_materials'
        
        # Folder path in the bucket
        self.path = 'gee_data_TFRecords/'
        
        # File number format
        self.nFormat = '{:02d}'
        
        
    def export_toCloudStorage(self):
        
        for n, point in enumerate(self.points):
            
            # Area of Interest
            geom = ee.Geometry.Point(point).buffer(self.buffer)
            region = geom.bounds().getInfo()['coordinates']
            
        
            for nC, collection in enumerate(self.collections):
            
                # Image Collection
                image_collection = ee_collection_specifics_TFRecord.ee_collections(collection)
 
                # Bands
                bands = ee_collection_specifics_TFRecord.ee_bands(collection)
    
                # Normalized Difference bands
                normDiff_bands = ee_collection_specifics_TFRecord.normDiff_bands(collection)
        
                # Normalized Difference band names
                normDiff_band_names = ee_collection_specifics_TFRecord.normDiff_band_names(collection)
                
        
                ## Composite
                image = ee_collection_specifics_TFRecord.Composite(collection)(image_collection, self.startDate, self.stopDate, geom)
        
                ## Calculate normalized Difference
                if normDiff_bands:
                    for nB, normDiff_band in enumerate(normDiff_bands):
                        image_nd = image.normalizedDifference(normDiff_band).rename(normDiff_band_names[nB])
                        ## Concatenate images into one multi-band image
                        if nB == 0:
                            image = ee.Image.cat([image.select(bands), image_nd])
                        else:
                            image = ee.Image.cat([image, image_nd])
                else:
                    image = image.select(bands)
                    
                    
                ## Change data type
                if ee_collection_specifics_TFRecord.Dtype(collection):
                    image = ee_collection_specifics_TFRecord.Dtype(collection)(image)
                    
                ## Concatenate images from different collections
                if nC == 0:
                    images = image
                else:
                    images = ee.Image.cat([images, image])
           
            ## Choose the scale
            images =  images.reproject(crs='EPSG:4326', scale=self.scale)
            
            ## Export image to Google Cloud Storage
            ee.batch.Export.image.toCloudStorage(
                image = images,
                description = 'Exporting '+self.file_name+'_'+self.nFormat.format(n),
                bucket= self.bucket,
                fileNamePrefix = self.path+self.file_name+'_'+self.nFormat.format(n),
                scale = self.scale,
                crs = 'EPSG:4326',
                region = region,
                maxPixels = 1e13,
                fileDimensions = [2560000,2560000],
                fileFormat= 'TFRecord',
                formatOptions= {'patchDimensions': self.patch_size,
                                'compressed': True}).start()
            

            

                
                
           
            
            