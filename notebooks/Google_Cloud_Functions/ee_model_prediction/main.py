
import ee
import json
import numpy as np
import requests
import ee_collection_specifics

account = 'skydipper@skydipper-196010.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(account, 'privatekey.json')
ee.Initialize(credentials)

def output_image(image, outBands, project_id, model_name, version_name, scale):
    # Load the trained model and use it for prediction.
    model = ee.Model.fromAiPlatformPredictor(
        projectName = project_id,
        modelName = model_name,
        version = version_name,
        inputTileSize = [144, 144],
        inputOverlapSize = [8, 8],
        proj = ee.Projection('EPSG:4326').atScale(scale),
        fixInputProj = True,
        outputBands = {'prediction': {
            'type': ee.PixelType.float(),
            'dimensions': 1,
          }                  
        }
    )
    prediction = model.predictImage(image.toArray()).arrayFlatten([outBands])
    
    return prediction

def ee_tile_url(image, collection, bands):
    # Define the URL format used for Earth Engine generated map tiles.
    EE_TILES = 'https://earthengine.googleapis.com/map/{mapid}/{{z}}/{{x}}/{{y}}?token={token}'
    
    dic = {}
    for params in ee_collection_specifics.vizz_params(collection):
        result =  all(elem in bands for elem in params.get('bands'))
        if result:
            mapid = image.getMapId(params)
            
            dic['tile_url_'+str(params.get('bands'))] = EE_TILES.format(**mapid)
    
    return dic

def ee_model_prediction(request):
    request = request.get_json()

    # Variables
    inComposite = request.get('composite') 
    inCollection = request.get('in_collection')
    outCollection = request.get('out_collection')
    inBands = request.get('in_bands')
    outBands = request.get('out_bands')
    startDate = request.get('start')
    stopDate = request.get('end')
    scale = request.get('scale')
    project_id = request.get('project_id')
    model_name = request.get('model_name')
    version_name = request.get('version_name')
    geometry = request.get('geometry')

    # Get input imagery on which it was trained the model
    image = ee.deserializer.fromJSON(inComposite)
    # Select bands and convert them into float
    image = image.select(inBands).float()
        
    # Get output imagery 
    prediction = output_image(image, outBands, project_id, model_name, version_name, scale)
    
    # Clip the prediction area with the polygon
    polygon = ee.Geometry.Polygon(geometry.get('features')[0].get('geometry').get('coordinates'))
    prediction = prediction.clip(polygon)

    # Get centroid
    centroid = polygon.centroid().getInfo().get('coordinates')[::-1]
    
    # Output
    output = {}
    output.update({"centroid": centroid})   
    # Input bands tile urls
    output.update(ee_tile_url(image, inCollection, inBands))
    # Output bands tile urls
    output.update(ee_tile_url(prediction, outCollection, outBands))
    
    # Serialize prediction 
    output.update({'prediction': prediction.serialize()})
    
    return json.dumps(output)