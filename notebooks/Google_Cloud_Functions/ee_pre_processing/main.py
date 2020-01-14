
import ee
import json
import numpy as np
import ee_collection_specifics
import env

account = env.service_account
credentials = ee.ServiceAccountCredentials(account, 'privatekey.json')
ee.Initialize(credentials)

def min_max_values(image, collection, scale):
    
    normThreshold = ee_collection_specifics.ee_bands_normThreshold(collection)
    
    num = 2
    lon = np.linspace(-180, 180, num)
    lat = np.linspace(-90, 90, num)
    
    features = []
    for i in range(len(lon)-1):
        for j in range(len(lat)-1):
            features.append(ee.Feature(ee.Geometry.Rectangle(lon[i], lat[j], lon[i+1], lat[j+1])))
    
    regReducer = {
        'geometry': ee.FeatureCollection(features),
        'reducer': ee.Reducer.minMax(),
        'maxPixels': 1e10,
        'bestEffort': True,
        'scale':scale
        
    }
    
    values = image.reduceRegion(**regReducer).getInfo()
    print(values)
    
    # Avoid outliers by taking into account only the normThreshold% of the data points.
    regReducer = {
        'geometry': ee.FeatureCollection(features),
        'reducer': ee.Reducer.histogram(),
        'maxPixels': 1e10,
        'bestEffort': True,
        'scale':scale
        
    }
    
    hist = image.reduceRegion(**regReducer).getInfo()

    for band in list(normThreshold.keys()):
        if normThreshold[band] != 100:
            count = np.array(hist.get(band).get('histogram'))
            x = np.array(hist.get(band).get('bucketMeans'))
        
            cumulative_per = np.cumsum(count/count.sum()*100)
        
            values[band+'_max'] = x[np.where(cumulative_per < normThreshold[band])][-1]
        
    return values

def normalize_ee_images(image, collection, values):
    
    Bands = ee_collection_specifics.ee_bands(collection)
       
    # Normalize [0, 1] ee images
    for i, band in enumerate(Bands):
        if i == 0:
            image_new = image.select(band).clamp(values[band+'_min'], values[band+'_max'])\
                                .subtract(values[band+'_min'])\
                                .divide(values[band+'_max']-values[band+'_min'])
        else:
            image_new = image_new.addBands(image.select(band).clamp(values[band+'_min'], values[band+'_max'])\
                                    .subtract(values[band+'_min'])\
                                    .divide(values[band+'_max']-values[band+'_min']))
            
    return image_new
    
def ee_pre_processing(request):
    request = request.get_json()

    # Variables
    collection = request.get('collection')
    startDate = ee.Date(request.get('start'))
    stopDate  = ee.Date(request.get('end'))
    scale  = request.get('scale')
    # Bands
    bands = ee_collection_specifics.ee_bands(collection)

    # Get composite
    image = ee_collection_specifics.Composite(collection)(startDate, stopDate)
    image = image.select(bands)

    # Normalize images
    if ee_collection_specifics.normalize(collection):
        # Get min man values for each band
        values = min_max_values(image, collection, scale)

        # Normalize images
        image = normalize_ee_images(image, collection, values)
    else:
        values = {}
        
    return json.dumps({'bands_min_max': values, 'composite': image.serialize()})
