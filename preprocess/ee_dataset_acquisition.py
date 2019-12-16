from urllib.request import urlopen
import zipfile
import gzip
import rasterio
import os, urllib
import sys
import shutil
import numpy as np
import math
from preprocess import ee_collection_specifics
import ee
import h5py

def subfield(cube, xr, yr):
    #Subfield selection
    cube_sub = cube[yr[0]:yr[1],xr[0]:xr[1],:]
    return cube_sub

class ee_dataset:
    
    def __init__(self, points, buffer, startDate, stopDate, scale, file_name, dataset_name, chunk_size, collection):
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
        file_name: string
            File name prefix.
        dataset_name: string
            h5py dataset name.
        chunk_size: tuple
            tuple with the (y,x) chunk size.
        collection: string
            Name of each collection.

        """
        
        self.points = points
        self.buffer = buffer
        self.startDate = startDate
        self.stopDate = stopDate       
        self.scale = scale 
        self.file_name = file_name 
        self.dataset_name = dataset_name
        self.chunk_size = chunk_size
        self.collection = collection
        
        # Number of points
        self.nPoints=len(self.points)
        
        # Image Collection
        self.image_collection = ee_collection_specifics.ee_collections(self.collection)
 
        # Bands
        self.bands = ee_collection_specifics.ee_bands(self.collection)
    
        # Normalized Difference bands
        self.normDiff_bands = ee_collection_specifics.normDiff_bands(self.collection)
        
        # Normalized Difference band names
        self.normDiff_band_names = ee_collection_specifics.normDiff_band_names(self.collection)
        
        # h5py file dtype
        self.h5py_dtype = ee_collection_specifics.h5py_dtype(self.collection)
        
        # Google Cloud Bucket
        self.bucket = 'skydipper_materials'
        
        # Folder path in the bucket
        self.path = 'gee_data/'
        
        # File number format
        self.nFormat = '{:02d}'
        
        # Path of the output files.
        self.root_out = './samples/'
        
    def export_toCloudStorage(self):
        
        for n, point in enumerate(self.points):
            
            # Area of Interest
            geom = ee.Geometry.Point(point).buffer(self.buffer)
            region = geom.bounds().getInfo()['coordinates']
        
            ## Composite
            image = ee_collection_specifics.Composite(self.collection)(self.image_collection, self.startDate, self.stopDate, geom)
        
            ## Calculate normalized Difference
            if self.normDiff_bands:
                for nB, normDiff_band in enumerate(self.normDiff_bands):
                    image_nd = image.normalizedDifference(normDiff_band).rename(self.normDiff_band_names[nB])
                    ## Concatenate images into one multi-band image
                    if nB == 0:
                        image = ee.Image.cat([image.select(self.bands), image_nd])
                    else:
                        image = ee.Image.cat([image, image_nd])
            else:
                image = image.select(self.bands)
           
            ## Choose the scale
            image =  image.reproject(crs='EPSG:4326', scale=self.scale)
            
            ## Change data type
            if ee_collection_specifics.Dtype(self.collection):
                image = ee_collection_specifics.Dtype(self.collection)(image)
            
            ## Export image to Google Cloud Storage
            ee.batch.Export.image.toCloudStorage(
                image = image,
                description = 'Exporting_'+self.file_name+'_'+self.nFormat.format(n),
                bucket= self.bucket,
                fileNamePrefix = self.path+self.file_name+'_'+self.nFormat.format(n),
                scale = self.scale,
                crs = 'EPSG:4326',
                region = region,
                maxPixels = 1e10,
                fileFormat= 'GeoTIFF',
                formatOptions= {'cloudOptimized': True}).start()
            
    def read_fromCloudStorage(self):
        
        for n in range(self.nPoints):
            ## File path
            filepath = f'https://storage.googleapis.com/{self.bucket}/{self.path}{self.file_name}'+'_'+self.nFormat.format(n)+'.tif'

            ## Read image with rasterio
            with rasterio.open(filepath) as image:
                
                nBands = image.count
                szy = image.height
                szx = image.width
            
                ## Save image with h5py in chunks
                with h5py.File(self.root_out+self.dataset_name+'_'+self.nFormat.format(n)+'.hdf5', 'w') as f:
                    data = f.create_dataset(self.dataset_name+'_'+self.nFormat.format(n), (szy,szx,nBands), chunks=True, dtype=self.h5py_dtype)
            
                    for nB in range(nBands):
                        data[:,:,nB] = image.read(nB+1)
                                 

    def resize_inChunks(self):
        
        cy, cx = self.chunk_size[0], self.chunk_size[1]
        
        for n in range(self.nPoints):
            
            with h5py.File(self.root_out+self.dataset_name+'_'+self.nFormat.format(n)+'.hdf5', 'r') as f:
                data = f[self.dataset_name+'_'+self.nFormat.format(n)]
            
                sy, sx, sz = data.shape
    
                num_pathces_per_frame = math.floor(sy/cy)*math.floor(sx/cx)
    
                ## Save image with h5py 
                with h5py.File(self.root_out+self.dataset_name+'_chunk_'+self.nFormat.format(n)+'.hdf5', 'w') as f:
                    data_new = f.create_dataset(self.dataset_name+'_chunk_'+self.nFormat.format(n), (int(num_pathces_per_frame),cy,cx,int(sz)), dtype=self.h5py_dtype)

                    nc=0
                    for j in np.arange(math.floor(sy/cy)):
                        for i in np.arange(math.floor(sx/cx)):

                            yr=[int(cy*j),int(cy+cy*j)]
                            xr=[int(cx*i),int(cx+cx*i)]
            
                            data_new[nc,:,:,:] = subfield(data,xr,yr)

                            nc+=1
                    
            ## Remove input file
            os.remove(self.root_out+self.dataset_name+'_'+self.nFormat.format(n)+'.hdf5')
        
        
    def merge_datasets(self):
        
        cy, cx = self.chunk_size[0], self.chunk_size[1]    
        
        nFrames = 0
        for n in range(self.nPoints):
            
            with h5py.File(self.root_out+self.dataset_name+'_chunk_'+self.nFormat.format(n)+'.hdf5', 'r') as f:
                data = f[self.dataset_name+'_chunk_'+self.nFormat.format(n)]
                
                nBands = data.shape[3]
                
                nFrames+=data.shape[0]
                
        ## Save image with h5py 
        with h5py.File(self.root_out+self.dataset_name+'.hdf5', 'w') as f:
            data_new = f.create_dataset(self.dataset_name, (nFrames,cy,cx,nBands), dtype=self.h5py_dtype)
            
            nFrames = 0
            for n in range(self.nPoints):
            
                with h5py.File(self.root_out+self.dataset_name+'_chunk_'+self.nFormat.format(n)+'.hdf5', 'r') as f:
                    data = f[self.dataset_name+'_chunk_'+self.nFormat.format(n)]
                    
                    data_new[nFrames:(nFrames+data.shape[0]),:,:,:] = data[:]
                    
                    nFrames+=data.shape[0]
                    
                    
                ## Remove input file
                os.remove(self.root_out+self.dataset_name+'_chunk_'+self.nFormat.format(n)+'.hdf5')
            

                
                
           
            
            