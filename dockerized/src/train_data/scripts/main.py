print("Starting process")
print("Importing packages")
import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import h5py
import os
import time
import math

print("Initializing GEE")
ee.Initialize()
print("Done")

# Central position of (AOIs)
print("Setting AOIs")
points = [[-120.7224, 37.3872], [-112.6799, 42.9816], [-89.7649, 35.8764], 
          [-96.0181, 41.2412], [-115.473, 46.861], [-103.9803, 47.9713], 
          [-96.9217, 32.8958], [-82.986, 40.019], [-90.347, 38.668], 
          [-110.6947, 37.4568], [-101.8889, 33.5527], [-92.621, 33.417],
          [-80.352, 38.628], [-104.752, 43.972], [-110.92, 37.18]]

# Start and stop of time series
print("Setting dates")
startDate = ee.Date('2016-01-01')
stopDate  = ee.Date('2016-12-31')
# Scale in meters
print("Setting scale and buffer size")
scale = 10
# Buffer
buffer = 40000

from preprocess.ee_dataset_acquisition import ee_dataset

print("Exporting tiffiles to Google Cloud Storage")
print("Exporting sentinel files")
sentinel = ee_dataset(points = points, buffer = buffer, startDate = startDate, stopDate = stopDate, scale = scale, file_name='S2_AOI', dataset_name = 'data_x', chunk_size = (128,128), collection = 'Sentinel2')
sentinel.export_toCloudStorage()
print("Exporting cropland files")
cropland = ee_dataset(points = points, buffer = buffer, startDate = startDate, stopDate = stopDate, scale = scale, file_name='cropland_AOI', dataset_name = 'data_y', chunk_size = (128,128), collection = 'CroplandDataLayers')
cropland.export_toCloudStorage()

print("Done")
