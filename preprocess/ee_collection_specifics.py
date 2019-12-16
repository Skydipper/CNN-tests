"""
Information on Earth Engine collections stored here (e.g. bands, collection ids, etc.)
"""

import ee
import numpy as np
import pandas as pd
from keras.utils import np_utils

def ee_bands(collection):
    """
    Earth Engine band names
    """
    
    dic = {
        'Sentinel2': ['B4','B3','B2','B8'],
        'Landsat7': ['B3','B2','B1','B4'],
        'CroplandDataLayers': ['cropland']
    }
    
    return dic[collection]

def ee_bands_rgb(collection):
    """
    Earth Engine band names
    """
    
    dic = {
        'Sentinel2': ['B4','B3','B2'],
        'Landsat7': ['B3','B2','B1'],
        'CroplandDataLayers': ['cropland']
    }
    
    return dic[collection]

def ee_collections(collection):
    """
    Earth Engine image collection names
    """
    dic = {
        'Sentinel2': 'COPERNICUS/S2',
        'Landsat7': 'LANDSAT/LE07/C01/T1_SR',
        'CroplandDataLayers': 'USDA/NASS/CDL'
    }
    
    return dic[collection]

def normDiff_bands(collection):
    """
    Earth Engine normDiff bands
    """
    dic = {
        'Sentinel2': [['B8','B4'], ['B8','B3']],
        'Landsat7': [['B4','B3'], ['B4','B2']],
        'CroplandDataLayers': []
    }
    
    return dic[collection]

def normDiff_band_names(collection):
    """
    Earth Engine normDiff bands
    """
    dic = {
        'Sentinel2': ['ndvi', 'ndwi'],
        'Landsat7': ['ndvi', 'ndwi'],
        'CroplandDataLayers': []
    }
    
    return dic[collection]

def vizz_params(collection):
    """
    Visualization parameters
    """
    dic = {
        'Sentinel2': [{'min':0,'max':150, 'bands':['B4','B3','B2']}, 
                      {'min':0,'max':150, 'bands':['B8']},
                      {'min':0,'max':200, 'bands':['ndvi']},
                      {'min':0,'max':200, 'bands':['ndwi']}],
        'Landsat7': [{'min':0,'max':150, 'bands':['B3','B2','B1']}, 
                      {'min':0,'max':200, 'bands':['B4']},
                      {'min':0,'max':200, 'bands':['ndvi']},
                      {'min':0,'max':200, 'bands':['ndwi']}],
        'CroplandDataLayers': [{'min':0,'max':255, 'bands':['cropland']}]
    }
    
    return dic[collection]

## ------------------------- Filter datasets ------------------------- ##
## Lansat 7 Cloud Free Composite
def CloudMaskL7(image):
    qa = image.select('pixel_qa')
    #If the cloud bit (5) is set and the cloud confidence (7) is high
    #or the cloud shadow bit is set (3), then it's a bad pixel.
    cloud = qa.bitwiseAnd(1 << 5).And(qa.bitwiseAnd(1 << 7)).Or(qa.bitwiseAnd(1 << 3))
    #Remove edge pixels that don't occur in all bands
    mask2 = image.mask().reduce(ee.Reducer.min())
    return image.updateMask(cloud.Not()).updateMask(mask2)

def CloudFreeCompositeL7(Collection_id, startDate, stopDate, geom):
    ## Define your collection
    collection = ee.ImageCollection(Collection_id)

    ## Filter 
    collection = collection.filterBounds(geom).filterDate(startDate,stopDate)\
            .map(CloudMaskL7)

    ## Composite
    composite = collection.median()
    
    return composite

## Sentinel 5 Cloud Free Composite
def CloudMaskS2(image):
    """
    European Space Agency (ESA) clouds from 'QA60', i.e. Quality Assessment band at 60m
    parsed by Nick Clinton
    """
    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = int(2**10)
    cirrusBitMask = int(2**11)

    # Both flags set to zero indicates clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(\
            qa.bitwiseAnd(cirrusBitMask).eq(0))

    return image.updateMask(mask).divide(10000)

def CloudFreeCompositeS2(Collection_id, startDate, stopDate, geom):
    ## Define your collection
    collection = ee.ImageCollection(Collection_id)

    ## Filter 
    collection = collection.filterBounds(geom).filterDate(startDate,stopDate)\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
            .map(CloudMaskS2)

    ## Composite
    composite = collection.median()
    
    return composite

## Cropland Data Layers
def CroplandData(Collection_id, startDate, stopDate, geom):
    ## Define your collection
    collection = ee.ImageCollection(Collection_id)

    ## Filter 
    collection = collection.filterBounds(geom).filterDate(startDate,stopDate)

    ## First image
    image = ee.Image(collection.first())
    
    return image

## ------------------------------------------------------------------- ##

def Composite(collection):
    dic = {
        'Sentinel2': CloudFreeCompositeS2,
        'Landsat7': CloudFreeCompositeL7,
        'CroplandDataLayers': CroplandData
    }
    
    return dic[collection]

## ------------------------ Change data types ------------------------ ##

def DataTypeS2(image):

    ## Change RGB data type to unsigned int8.
    RGB = image.expression('(400 * RGB)', {'RGB': image.select(['B4', 'B3', 'B2'])})

    RGB = RGB.byte()

    ## Change NIR data type to unsigned int16.
    NIR = image.expression('(400 * NIR)', {'NIR': image.select(['B8'])})

    NIR = NIR.byte().rename('B8')

    ## Change NDVI and NDWI data types to unsigned int8.
    NDVI = image.expression('(300 * NDVI)', {'NDVI': image.select(['ndvi'])});
      
    NDVI = NDVI.byte().rename('ndvi')
    
    NDWI = image.expression('(300 * NDWI)', {'NDWI': image.select(['ndwi'])});

    NDWI = NDWI.byte().rename('ndwi')
    
    ## Concatenate Bands with new data types
    image = ee.Image.cat([RGB, NIR, NDVI, NDWI])

    return image

def DataTypeL7(image):

    ## Change RGB data type to unsigned int8.
    RGB = image.expression('(RGB/20)', {'RGB': image.select(['B3', 'B2', 'B1'])})

    RGB = RGB.byte()

    ## Change NIR data type to unsigned int16.
    NIR = image.expression('(NIR/20)', {'NIR': image.select(['B4'])})

    NIR = NIR.byte().rename('B4')

    ## Change NDVI and NDWI data types to unsigned int8.
    NDVI = image.expression('(300 * NDVI)', {'NDVI': image.select(['ndvi'])});
      
    NDVI = NDVI.byte().rename('ndvi')
    
    NDWI = image.expression('(300 * NDWI)', {'NDWI': image.select(['ndwi'])});

    NDWI = NDWI.byte().rename('ndwi')
    
    ## Concatenate Bands with new data types
    image = ee.Image.cat([RGB, NIR, NDVI, NDWI])

    return image
## ------------------------------------------------------------------- ##

def Dtype(collection):
    dic = {
        'Sentinel2': DataTypeS2,
        'Landsat7': DataTypeL7,
        'CroplandDataLayers': []
    }
    
    return dic[collection]

def h5py_dtype(collection):
    dic = {
        'Sentinel2': np.uint8,
        'Landsat7': np.uint8,
        'CroplandDataLayers': np.uint8
    }
    
    return dic[collection]


## ------------------------ Change class labels ------------------------ ##

def replace_values(array, class_labels, new_label):
    array_new = np.copy(array)
    for i in range(len(class_labels)):
        array_new[np.where(array == class_labels[i])] = new_label
        
    return array_new

def cropland_classes(data):
    # Area of Interest (AoI)
    point = [-120.7224, 37.3872]
    geom = ee.Geometry.Point(point).buffer(100)
    # Start and stop of time series
    startDate = ee.Date('2016')
    stopDate  = ee.Date('2017')
    # Read the ImageCollection
    dataset = ee.ImageCollection('USDA/NASS/CDL')\
        .filterBounds(geom)\
        .filterDate(startDate,stopDate)
    # Get the cropland class values and names
    cropland_info = pd.DataFrame({'cropland_class_values':dataset.getInfo().get('features')[0].get('properties').get('cropland_class_values'),
                              'cropland_class_palette':dataset.getInfo().get('features')[0].get('properties').get('cropland_class_palette'),
                              'cropland_class_names':dataset.getInfo().get('features')[0].get('properties').get('cropland_class_names')
                             })
    

    # New classes
    land = ['Shrubland', 'Barren', 'Grassland/Pasture', 'Deciduous Forest', 'Evergreen Forest', 'Mixed Forest', 'Wetlands', 'Woody Wetlands', 'Herbaceous Wetlands']
    water = ['Water', 'Open Water', 'Aquaculture']
    urban = ['Developed', 'Developed/Open Space', 'Developed/High Intensity', 'Developed/Low Intensity', 'Developed/Med Intensity']

    class_labels_0 = np.array(cropland_info[cropland_info['cropland_class_names'].isin(land)]['cropland_class_values'])
    class_labels_1 = np.array(cropland_info[cropland_info['cropland_class_names'].isin(water)]['cropland_class_values'])
    class_labels_2 = np.array(cropland_info[cropland_info['cropland_class_names'].isin(urban)]['cropland_class_values'])
    class_labels_3 = np.array(cropland_info[(~cropland_info['cropland_class_names'].isin(land)) & 
                                        (~cropland_info['cropland_class_names'].isin(water)) & 
                                        (~cropland_info['cropland_class_names'].isin(urban))]['cropland_class_values'])

    # We replace the class labels
    new_data = np.copy(data[:])
    new_data = replace_values(new_data, class_labels_3, 3)
    new_data = replace_values(new_data, class_labels_2, 2)
    new_data = replace_values(new_data, class_labels_1, 1)
    new_data = replace_values(new_data, class_labels_0, 0)

    # Convert 1-dimensional class arrays to 4-dimensional class matrices
    new_data = np_utils.to_categorical(new_data, 4)
    
    return new_data


## ------------------------------------------------------------------- ##

def change_class_labels(collection):
    dic = {
        'Sentinel2': [],
        'Landsat7': [],
        'CroplandDataLayers': cropland_classes
    }
    
    return dic[collection]

def nClasses(collection):
    dic = {
        'Sentinel2': [],
        'Landsat7': [],
        'CroplandDataLayers': 4
    }
    
    return dic[collection]