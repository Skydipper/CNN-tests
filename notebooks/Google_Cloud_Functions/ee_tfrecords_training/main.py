
import ee
import json
import numpy as np
import requests
import ee_collection_specifics
import env

account = env.service_account
credentials = ee.ServiceAccountCredentials(account, 'privatekey.json')
ee.Initialize(credentials)

def image_into_array(url, collections, bands, kernelSize, startDate, stopDate, scale):

    headers = {'Content-Type': 'application/json'}
    
    for i, collection in enumerate(collections):
        payload =   {
            "collection": collection,
            "start": startDate,
            "end": stopDate,
            "scale": scale
        }
        
        output = requests.post(url, data=json.dumps(payload), headers=headers)
        
        if i == 0:
            image = ee.deserializer.fromJSON(output.json()['composite']).select(bands[i])
        else:
            featureStack = ee.Image.cat([image,\
                                         ee.deserializer.fromJSON(output.json()['composite']).select(bands[i])\
                                        ]).float()
            
    list = ee.List.repeat(1, kernelSize)
    lists = ee.List.repeat(list, kernelSize)
    kernel = ee.Kernel.fixed(kernelSize, kernelSize, lists)
    
    arrays = featureStack.neighborhoodToArray(kernel)
    
    return arrays

def GeoJSONs_to_FeatureCollections(multipolygon):
    # Make a list of Features
    features = []
    for i in range(len(multipolygon.get('features')[0].get('geometry').get('coordinates'))):
        features.append(
            ee.Feature(
                ee.Geometry.Polygon(
                    multipolygon.get('features')[0].get('geometry').get('coordinates')[i]
                )
            )
        )
        
    # Create a FeatureCollection from the list and print it.
    return ee.FeatureCollection(features)

def export_TFRecords(arrays, scale, nShards, sampleSize, features, polysLists, baseNames, bucket, folder, selectors):
    # Export all the training/evaluation data (in many pieces), with one task per geometry.
    filePaths = []
    for i, feature in enumerate(features):
        for g in range(feature.size().getInfo()):
            geomSample = ee.FeatureCollection([])
            for j in range(nShards):
                sample = arrays.sample(
                    region = ee.Feature(polysLists[i].get(g)).geometry(), 
                    scale = scale, 
                    numPixels = sampleSize / nShards, # Size of the shard.
                    seed = j,
                    tileScale = 8
                )
                geomSample = geomSample.merge(sample)
                
            desc = baseNames[i] + '_g' + str(g)
            
            filePaths.append(bucket+ '/' + folder + '/' + desc)
            
            task = ee.batch.Export.table.toCloudStorage(
                collection = geomSample,
                description = desc, 
                bucket = bucket, 
                fileNamePrefix = folder + '/' + desc,
                fileFormat = 'TFRecord',
                selectors = selectors
            )
            task.start()
            
    return filePaths   

def ee_tfrecords_training(request):
    request = request.get_json()

    # Variables
    inCollection = request.get('in_collection')
    outCollection = request.get('out_collection')
    inBands = request.get('in_bands')
    outBands = request.get('out_bands')
    startDate = request.get('start')
    stopDate = request.get('end')
    scale = request.get('scale')
    sampleSize = request.get('sample_size')
    datasetName = request.get('dataset_name')
    trainPolys = request.get('train_polys')
    evalPolys = request.get('eval_polys')

    # An array of images
    url = f'https://us-central1-skydipper-196010.cloudfunctions.net/ee_pre_processing'
    collections = [inCollection, outCollection]
    bands = [inBands, outBands]
    kernelSize = 256

    arrays = image_into_array(url, collections, bands, kernelSize, startDate, stopDate, scale)
    
    # Convert the GeoJSONs to feature collections
    trainFeatures = GeoJSONs_to_FeatureCollections(trainPolys)
    evalFeatures = GeoJSONs_to_FeatureCollections(evalPolys)

    # Convert the feature collections to lists for iteration.
    trainPolysList = trainFeatures.toList(trainFeatures.size())
    evalPolysList = evalFeatures.toList(evalFeatures.size())

    # These numbers determined experimentally.
    nShards  = int(sampleSize/20)#100 # Number of shards in each polygon.

    features = [trainFeatures, evalFeatures]
    polysLists = [trainPolysList, evalPolysList]
    baseNames = ['training_patches', 'eval_patches']
    bucket = 'skydipper_materials'
    folder = 'cnn-models/'+datasetName+'/data'
    selectors = inBands + outBands

    # Export all the training/evaluation data (in many pieces), with one task per geometry.
    filePaths   = export_TFRecords(arrays, scale, nShards, sampleSize, features, polysLists, baseNames, bucket, folder, selectors)    

        
    return json.dumps({
      "file_paths": filePaths, 
      "training_polygons": trainFeatures.serialize(), 
      "evaluation_polygons": evalFeatures.serialize()
      })