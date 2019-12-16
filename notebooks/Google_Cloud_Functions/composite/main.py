import ee
import json
import numpy as np
import ee_collection_specifics

account = 'skydipper@skydipper-196010.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(account, 'privatekey.json')
ee.Initialize(credentials)

def ThumbURL(image, viz_params=None):
    """Create a target url for tumb for an image.
    """
    if viz_params:
        url = image.getThumbURL(viz_params)
    else:
        url = image.getThumbURL()
    return url

def TileURL(image, viz_params=None):
    """Create a target url for tiles for an image.
    """
    if viz_params:
        d = image.getMapId(viz_params)
    else:
        d = image.getMapId()
    base_url = 'https://earthengine.googleapis.com'
    url = (base_url + '/map/' + d['mapid'] + "/{z}/{x}/{y}?token="+ d['token'])
    return url
    
def composite(request):
    request = request.get_json()

    # Geometry
    lon = request.get('lon')
    lat = request.get('lat')
    # Instrument
    collection = request.get('instrument')
    # Start and stop of time series
    startDate = ee.Date(request.get('start'))
    stopDate  = ee.Date(request.get('end'))
    
    ## Area of Interest
    point = ee.Geometry.Point([lon, lat]).buffer(1000)
    # bounding box
    coordinates = np.array(point.bounds().getInfo()['coordinates'][0])
    bbox = [min(coordinates[:,0]), min(coordinates[:,1]), max(coordinates[:,0]), max(coordinates[:,1])]
    # Rectangle
    geom = ee.Geometry.Rectangle(bbox)
    region = geom.bounds().getInfo()['coordinates']

    # Bands
    bands = ee_collection_specifics.ee_bands_rgb(collection)
    # Visualiztion parameters
    visParam = ee_collection_specifics.vizz_params_rgb(collection)

    # Image Collection
    image_collection = ee_collection_specifics.ee_collections(collection) 
    ## Composite
    image = ee_collection_specifics.Composite(collection)(image_collection, startDate, stopDate, geom)
        
    ## Select bands
    image = image.select(bands)

    ## Get ThumbURL
    thumb_url = ThumbURL(image, {'min':visParam['min'],'max':visParam['max'], 'region': region, 'dimensions': [256,256]})

    ## Get TileURL
    tile_url = TileURL(image.clip(geom), {'min':visParam['min'],'max':visParam['max']})
        
    return json.dumps({'thumb_url': thumb_url, 'tile_url': tile_url, 'bbox': bbox})