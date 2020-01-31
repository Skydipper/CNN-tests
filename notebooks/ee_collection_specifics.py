
"""
Information on Earth Engine collections stored here (e.g. bands, collection ids, etc.)
"""

import ee

def ee_collections(collection):
    """
    Earth Engine image collection names
    """
    dic = {
        'Sentinel-2-Top-of-Atmosphere-Reflectance': 'COPERNICUS/S2',
        'Landsat-7-Surface-Reflectance': 'LANDSAT/LE07/C01/T1_SR',
        'Landsat-8-Surface-Reflectance': 'LANDSAT/LC08/C01/T1_SR',
        'USDA-NASS-Cropland-Data-Layers': 'USDA/NASS/CDL',
        'USGS-National-Land-Cover-Database': 'USGS/NLCD',
        'Lake-Water-Quality-100m': 'projects/vizzuality/skydipper-water-quality/LWQ-100m'
    }
    
    return dic[collection]

def ee_bands(collection):
    """
    Earth Engine band names
    """
    
    dic = {
        'Sentinel-2-Top-of-Atmosphere-Reflectance': ['B1','B2','B3','B4','B5','B6','B7','B8A','B8','B11','B12','ndvi','ndwi'],
        'Landsat-7-Surface-Reflectance': ['B1','B2','B3','B4','B5','B6','B7','ndvi','ndwi'],
        'Landsat-8-Surface-Reflectance': ['B1','B2','B3','B4','B5','B6','B7','B10','B11','ndvi','ndwi'],
        'USDA-NASS-Cropland-Data-Layers': ['landcover', 'cropland', 'land', 'water', 'urban'],
        'USGS-National-Land-Cover-Database': ['impervious'],
        'Lake-Water-Quality-100m': ['turbidity_blended_mean']
    }
    
    return dic[collection]

def ee_bands_rgb(collection):
    """
    Earth Engine rgb band names
    """
    
    dic = {
        'Sentinel-2-Top-of-Atmosphere-Reflectance': ['B4','B3','B2'],
        'Landsat-7-Surface-Reflectance': ['B3','B2','B1'],
        'Landsat-8-Surface-Reflectance': ['B4', 'B3', 'B2'],
        'USDA-NASS-Cropland-Data-Layers': ['landcover'],
        'USGS-National-Land-Cover-Database': ['impervious'],
        'Lake-Water-Quality-100m': ['turbidity_blended_mean']
    }
    
    return dic[collection]

def ee_bands_normThreshold(collection):
    """
    Normalization threshold percentage
    """
    
    dic = {
        'Sentinel-2-Top-of-Atmosphere-Reflectance': {'B1': 75,'B2': 75,'B3': 75,'B4': 75,'B5': 80,'B6': 80,'B7': 80,'B8A': 80,'B8': 80,'B11': 100,'B12': 100},
        'Landsat-7-Surface-Reflectance': {'B1': 95,'B2': 95,'B3': 95,'B4': 100,'B5': 100,'B6': 100,'B7': 100},
        'Landsat-8-Surface-Reflectance': {'B1': 90,'B2': 95,'B3': 95,'B4': 95,'B5': 100,'B6': 100,'B7': 100,'B10': 100,'B11': 100},
        'USDA-NASS-Cropland-Data-Layers': {'landcover': 100, 'cropland': 100, 'land': 100, 'water': 100, 'urban': 100},
        'USGS-National-Land-Cover-Database': {'impervious': 100},
        'Lake-Water-Quality-100m': {'turbidity_blended_mean': 100}
    }
    
    return dic[collection]

def normalize(collection):
    dic = {
        'Sentinel-2-Top-of-Atmosphere-Reflectance': True,
        'Landsat-7-Surface-Reflectance': True,
        'Landsat-8-Surface-Reflectance': True,
        'USDA-NASS-Cropland-Data-Layers': False,
        'USGS-National-Land-Cover-Database': False,
        'Lake-Water-Quality-100m': False
    }
    
    return dic[collection]

def vizz_params_rgb(collection):
    """
    Visualization parameters
    """
    dic = {
        'Sentinel-2-Top-of-Atmosphere-Reflectance': {'min':0,'max':3000, 'bands':['B4','B3','B2']},
        'Landsat-7-Surface-Reflectance': {'min':0,'max':3000, 'gamma':1.4, 'bands':['B3','B2','B1']},
        'Landsat-8-Surface-Reflectance': {'min':0,'max':3000, 'gamma':1.4, 'bands':['B4','B3','B2']},
        'USDA-NASS-Cropland-Data-Layers': {'min':0,'max':3, 'bands':['landcover']},
        'USGS-National-Land-Cover-Database': {'min': 0, 'max': 1, 'bands':['impervious']},
        'Lake-Water-Quality-100m': {'min': 0, 'max': 1, 'bands':['turbidity_blended_mean']}
    }
    
    return dic[collection]

def vizz_params(collection):
    """
    Visualization parameters
    """
    dic = {
        'Sentinel-2-Top-of-Atmosphere-Reflectance': [{'min':0,'max':1, 'bands':['B4','B3','B2']}, 
                      {'min':0,'max':1, 'bands':['B1']},
                      {'min':0,'max':1, 'bands':['B5']},
                      {'min':0,'max':1, 'bands':['B6']},
                      {'min':0,'max':1, 'bands':['B7']},
                      {'min':0,'max':1, 'bands':['B8A']},
                      {'min':0,'max':1, 'bands':['B8']},
                      {'min':0,'max':1, 'bands':['B11']},
                      {'min':0,'max':1, 'bands':['B12']},
                      {'min':0,'max':1, 'gamma':1.4, 'bands':['ndvi']},
                      {'min':0,'max':1, 'gamma':1.4, 'bands':['ndwi']}],
        'Landsat-7-Surface-Reflectance': [{'min':0,'max':1, 'gamma':1.4, 'bands':['B3','B2','B1']}, 
                     {'min':0,'max':1, 'gamma':1.4, 'bands':['B4']},
                     {'min':0,'max':1, 'gamma':1.4, 'bands':['B5']},
                     {'min':0,'max':1, 'gamma':1.4, 'bands':['B7']},
                     {'min':0,'max':1, 'gamma':1.4, 'bands':['B6']},
                     {'min':0,'max':1, 'gamma':1.4, 'bands':['ndvi']},
                     {'min':0,'max':1, 'gamma':1.4, 'bands':['ndwi']}],
        'Landsat-8-Surface-Reflectance': [{'min':0,'max':1, 'gamma':1.4, 'bands':['B4','B3','B2']}, 
                     {'min':0,'max':1, 'gamma':1.4, 'bands':['B1']},
                     {'min':0,'max':1, 'gamma':1.4, 'bands':['B5']},
                     {'min':0,'max':1, 'gamma':1.4, 'bands':['B6']},
                     {'min':0,'max':1, 'gamma':1.4, 'bands':['B7']},
                     {'min':0,'max':1, 'gamma':1.4, 'bands':['B10']},
                     {'min':0,'max':1, 'gamma':1.4, 'bands':['B11']},
                     {'min':0,'max':1, 'gamma':1.4, 'bands':['ndvi']},
                     {'min':0,'max':1, 'gamma':1.4, 'bands':['ndwi']}],
        'USDA-NASS-Cropland-Data-Layers': [{'min':0,'max':3, 'bands':['landcover']},
                               {'min':0,'max':1, 'bands':['cropland']},
                               {'min':0,'max':1, 'bands':['land']},
                               {'min':0,'max':1, 'bands':['water']},
                               {'min':0,'max':1, 'bands':['urban']}],
        'USGS-National-Land-Cover-Database': [{'min': 0, 'max': 1, 'bands':['impervious']}],
        'Lake-Water-Quality-100m': [{'min': 0, 'max': 1, 'bands':['turbidity_blended_mean']}],
        
    }
    
    return dic[collection]

## ------------------------- Filter datasets ------------------------- ##
## Lansat 7 Cloud Free Composite
def CloudMaskL7sr(image):
    qa = image.select('pixel_qa')
    #If the cloud bit (5) is set and the cloud confidence (7) is high
    #or the cloud shadow bit is set (3), then it's a bad pixel.
    cloud = qa.bitwiseAnd(1 << 5).And(qa.bitwiseAnd(1 << 7)).Or(qa.bitwiseAnd(1 << 3))
    #Remove edge pixels that don't occur in all bands
    mask2 = image.mask().reduce(ee.Reducer.min())
    return image.updateMask(cloud.Not()).updateMask(mask2)

def CloudFreeCompositeL7(startDate, stopDate):
    ## Define your collection
    collection = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')

    ## Filter 
    collection = collection.filterDate(startDate,stopDate).map(CloudMaskL7sr)

    ## Composite
    composite = collection.median()
    
    ## normDiff bands
    normDiff_band_names = ['ndvi', 'ndwi']
    for nB, normDiff_band in enumerate([['B4','B3'], ['B4','B2']]):
        image_nd = composite.normalizedDifference(normDiff_band).rename(normDiff_band_names[nB])
        composite = ee.Image.cat([composite, image_nd])
    
    return composite

## Lansat 8 Cloud Free Composite
def CloudMaskL8sr(image):
    opticalBands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    thermalBands = ['B10', 'B11']

    cloudShadowBitMask = ee.Number(2).pow(3).int()
    cloudsBitMask = ee.Number(2).pow(5).int()
    qa = image.select('pixel_qa')
    mask1 = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
    qa.bitwiseAnd(cloudsBitMask).eq(0))
    mask2 = image.mask().reduce('min')
    mask3 = image.select(opticalBands).gt(0).And(
            image.select(opticalBands).lt(10000)).reduce('min')
    mask = mask1.And(mask2).And(mask3)
    
    return image.updateMask(mask)

def CloudFreeCompositeL8(startDate, stopDate):
    ## Define your collection
    collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')

    ## Filter 
    collection = collection.filterDate(startDate,stopDate).map(CloudMaskL8sr)

    ## Composite
    composite = collection.median()
    
    ## normDiff bands
    normDiff_band_names = ['ndvi', 'ndwi']
    for nB, normDiff_band in enumerate([['B5','B4'], ['B5','B3']]):
        image_nd = composite.normalizedDifference(normDiff_band).rename(normDiff_band_names[nB])
        composite = ee.Image.cat([composite, image_nd])
    
    return composite

## Sentinel 2 Cloud Free Composite
def CloudMaskS2(image):
    """
    European Space Agency (ESA) clouds from 'QA60', i.e. Quality Assessment band at 60m
    parsed by Nick Clinton
    """
    AerosolsBands = ['B1']
    VIBands = ['B2', 'B3', 'B4']
    RedBands = ['B5', 'B6', 'B7', 'B8A']
    NIRBands = ['B8']
    SWIRBands = ['B11', 'B12']

    qa = image.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = int(2**10)
    cirrusBitMask = int(2**11)

    # Both flags set to zero indicates clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(\
            qa.bitwiseAnd(cirrusBitMask).eq(0))

    return image.updateMask(mask)

def CloudFreeCompositeS2(startDate, stopDate):
    ## Define your collection
    collection = ee.ImageCollection('COPERNICUS/S2')

    ## Filter 
    collection = collection.filterDate(startDate,stopDate)\
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
            .map(CloudMaskS2)

    ## Composite
    composite = collection.median()
    
    ## normDiff bands
    normDiff_band_names = ['ndvi', 'ndwi']
    for nB, normDiff_band in enumerate([['B8','B4'], ['B8','B3']]):
        image_nd = composite.normalizedDifference(normDiff_band).rename(normDiff_band_names[nB])
        composite = ee.Image.cat([composite, image_nd])
    
    return composite

## Cropland Data Layers
def CroplandData(startDate, stopDate):
    ## Define your collection
    collection = ee.ImageCollection('USDA/NASS/CDL')

    ## Filter 
    collection = collection.filterDate(startDate,stopDate)

    ## First image
    image = ee.Image(collection.first())
    
    ## Change classes
    land = ['65', '131', '141', '142', '143', '152', '176', '87', '190', '195']
    water = ['83', '92', '111']
    urban = ['82', '121', '122', '123', '124']
    
    classes = []
    for n, i in enumerate([land,water,urban]):
        a = ''
        for m, j in enumerate(i):
            if m < len(i)-1:
                a = a + 'crop == '+ j + ' || '
            else: 
                a = a + 'crop == '+ j
        classes.append('('+a+') * '+str(n+1))
    classes = ' + '.join(classes)
    
    image = image.expression(classes, {'crop': image.select(['cropland'])})
    
    image =image.rename('landcover')
    
    # Split image into 1 band per class
    names = ['cropland', 'land', 'water', 'urban']
    mask = image
    for i, name in enumerate(names):
        image = ee.Image.cat([image, mask.eq(i).rename(name)])
     
    return image

## National Land Cover Database
def ImperviousData(startDate, stopDate):
    ## Define your collection
    collection = ee.ImageCollection('USGS/NLCD')

    ## Filter 
    collection = collection.filterDate(startDate,stopDate)

    ## First image
    image = ee.Image(collection.first())
    
    ## Select impervious band
    image = image.select('impervious')
    
    ## Normalize to 1
    image = image.divide(100).float()
    
    return image

def WaterQuality(startDate, stopDate):
    ## Define your collection
    collection = ee.ImageCollection('projects/vizzuality/skydipper-water-quality/LWQ-100m')

    ## Filter 
    collection = collection.filterDate(startDate,stopDate)

    ## First image
    image = ee.Image(collection.first())
    
    ## Select impervious band
    image = image.select('turbidity_blended_mean')
    
    return image

## ------------------------------------------------------------------- ##

def Composite(collection):
    dic = {
        'Sentinel-2-Top-of-Atmosphere-Reflectance': CloudFreeCompositeS2,
        'Landsat-7-Surface-Reflectance': CloudFreeCompositeL7,
        'Landsat-8-Surface-Reflectance': CloudFreeCompositeL8,
        'USDA-NASS-Cropland-Data-Layers': CroplandData,
        'USGS-National-Land-Cover-Database': ImperviousData,
        'Lake-Water-Quality-100m': WaterQuality
    }
    
    return dic[collection]
