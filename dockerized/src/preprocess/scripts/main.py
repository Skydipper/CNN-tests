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

sentinel = ee_dataset(points = points, buffer = buffer, startDate = startDate, stopDate = stopDate, scale = scale, file_name='S2_AOI', dataset_name = 'data_x', chunk_size = (128,128), collection = 'Sentinel2')

cropland = ee_dataset(points = points, buffer = buffer, startDate = startDate, stopDate = stopDate, scale = scale, file_name='cropland_AOI', dataset_name = 'data_y', chunk_size = (128,128), collection = 'CroplandDataLayers')

print("Getting data from Cloud Storage")

sentinel.read_fromCloudStorage()
cropland.read_fromCloudStorage()

print("Resizing data to chunks")
sentinel.resize_inChunks()
cropland.resize_inChunks()

print("Merging datasets")
sentinel.merge_datasets()
cropland.merge_datasets()

print("Normalizing values")
from preprocess.preprocess_datasets import preprocess_datasets
preprocess = preprocess_datasets(dataset_names=['data_x','data_y'], collections=['Sentinel2', 'CroplandDataLayers'])
preprocess.normalization_values()

print("Changing class labels")
preprocess.change_class_labels()
print("Randomizing datasets")
preprocess.randomize_datasets()

print("Splitting data into training and validation sets")
preprocess.train_validation_split(val_size=20)

print("Done")
