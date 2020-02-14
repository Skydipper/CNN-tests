import numpy as np
import pandas as pd
from sqlalchemy import Column, Integer, BigInteger, Float, Text, String, Boolean, DateTime
from sqlalchemy.dialects.postgresql import JSON
import json
import ee

import ee_collection_specifics

def df_from_query(engine, table_name):
    """Read DataFrames from query"""
    queries = {
        "dataset": "SELECT * FROM dataset",
        "image": "SELECT * FROM image",
        "model": "SELECT * FROM model",
        "model_versions": "SELECT * FROM model_versions",
    } 
    
    try:
        if table_name in queries.keys():
            df = pd.read_sql(queries.get(table_name), con=engine).drop(columns='id')
            
        return df
    except:
        print("Table doesn't exist in database!") 


def df_to_db(df, engine, table_name):
    """Save DataFrames into database."""
    if table_name == "dataset":
        df.to_sql("dataset",
                       engine,
                       if_exists='replace',
                       schema='public',
                       index=True,
                       index_label='id',
                       chunksize=500,
                       dtype={"slug": Text,
                              "name": Text,
                              "bands": Text,
                              "bands": Text,
                              "provider": Text})
    if table_name == "image":
        df.to_sql("image",
                       engine,
                       if_exists='replace',
                       schema='public',
                       index=True,
                       index_label='id',
                       chunksize=500,
                       dtype={"dataset_id ": Integer,
                              "bands_selections": Text,
                              "scale": Float,
                              "init_date": Text,
                              "end_date": Text,
                              "bands_min_max": JSON,
                              "norm_type": Text})
    
    if table_name == "model":
        df.to_sql("model",
                       engine,
                       if_exists='replace',
                       schema='public',
                       index=True,
                       index_label='id',
                       chunksize=500,
                       dtype={"model_name": Text,
                              "model_type": Text,
                              "model_output": Text,
                              "model_description": Text,
                              "output_image_id": Integer})
    
    if table_name == "model_versions":
        df.to_sql("model_versions",
                       engine,
                       if_exists='replace',
                       schema='public',
                       index=True,
                       index_label='id',
                       chunksize=500,
                       dtype={"model_id": Integer,
                              "model_architecture": Text,
                              "input_image_id": Integer,
                              "output_image_id": Integer,
                              "geostore_id": Text,
                              "kernel_size": BigInteger,
                              "sample_size": BigInteger,
                              "training_params": JSON,
                              "version": BigInteger,
                              "data_status": Text,
                              "training_status": Text,
                              "eeified": Boolean,
                              "deployed": Boolean})

def df_to_csv(df, table_name):
    table_paths = {
        "dataset": 'Database/dataset.csv',
        "image": 'Database/image.csv',
        "model": 'Database/model.csv',
        "model_versions": 'Database/model_versions.csv',
    } 
    
    try:
        if table_name in table_paths.keys():
            df.to_csv(table_paths.get(table_name),sep=';', quotechar='\'',index=True, index_label='id')
    except:
        print("Incorrect table name!")

def polygons_to_geoStoreMultiPoligon(Polygons):
    Polygons = list(filter(None, Polygons))
    MultiPoligon = {}
    properties = ["training", "validation", "test"]
    features = []
    for n, polygons in enumerate(Polygons):
        multipoligon = []
        for polygon in polygons.get('features'):
            multipoligon.append(polygon.get('geometry').get('coordinates'))
            
        features.append({
            "type": "Feature",
            "properties": {"name": properties[n]},
            "geometry": {
                "type": "MultiPolygon",
                "coordinates":  multipoligon
            }
        }
        ) 
        
    MultiPoligon = {
        "geojson": {
            "type": "FeatureCollection", 
            "features": features
        }
    }

    return MultiPoligon

def get_geojson_string(geom):
    coords = geom.get('coordinates', None)
    if coords and not any(isinstance(i, list) for i in coords[0]):
        geom['coordinates'] = [coords]
    feat_col = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": geom}]}
    return json.dumps(feat_col)

def min_max_values(image, collection, scale, norm_type='global', geostore=None, values = {}):
    
    normThreshold = ee_collection_specifics.ee_bands_normThreshold(collection)
    
    if not norm_type == 'custom':
        if norm_type == 'global':
            num = 2
            lon = np.linspace(-180, 180, num)
            lat = np.linspace(-90, 90, num)
            
            features = []
            for i in range(len(lon)-1):
                for j in range(len(lat)-1):
                    features.append(ee.Feature(ee.Geometry.Rectangle(lon[i], lat[j], lon[i+1], lat[j+1])))
        
        if norm_type == 'geostore':
            try:
                #geostore = Skydipper.Geometry(id_hash=geostore_id)
                features = []
                for feature in geostore.get('geojson').get('features'):
                    features.append(ee.Feature(feature))
                
            except:
                print('Geostore is needed')
        
        regReducer = {
            'geometry': ee.FeatureCollection(features),
            'reducer': ee.Reducer.minMax(),
            'maxPixels': 1e10,
            'bestEffort': True,
            'scale':scale,
            'tileScale': 10
            
        }
        
        values = image.reduceRegion(**regReducer).getInfo()
        
        # Avoid outliers by taking into account only the normThreshold% of the data points.
        regReducer = {
            'geometry': ee.FeatureCollection(features),
            'reducer': ee.Reducer.histogram(),
            'maxPixels': 1e10,
            'bestEffort': True,
            'scale':scale,
            'tileScale': 10
            
        }
        
        hist = image.reduceRegion(**regReducer).getInfo()
    
        for band in list(normThreshold.keys()):
            if normThreshold[band] != 100:
                count = np.array(hist.get(band).get('histogram'))
                x = np.array(hist.get(band).get('bucketMeans'))
            
                cumulative_per = np.cumsum(count/count.sum()*100)
            
                values[band+'_max'] = x[np.where(cumulative_per < normThreshold[band])][-1]
    else:
        values = values
        
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

def get_image_ids(images, datasets, slugs, bands, scale, init_date, end_date, norm_type):
    image_ids = []
    for n, slug in enumerate(slugs):
        dataset_id = datasets[datasets['slug'] == slug].index[0]
        df = images[(images['dataset_id'] == dataset_id) & 
                    (images['bands_selections'] == str(bands[n])) & 
                    (images['scale'] == scale) & 
                    (images['init_date'] == init_date) & 
                    (images['end_date'] == end_date) & 
                    (images['norm_type'] == norm_type) 
                ].copy()
        image_ids.append(df.index[0])
    return image_ids

def GeoJSONs_to_FeatureCollections(geostore):
    feature_collections = []
    for n in range(len(geostore.get('geojson').get('features'))):
        # Make a list of Features
        features = []
        for i in range(len(geostore.get('geojson').get('features')[n].get('geometry').get('coordinates'))):
            features.append(
                ee.Feature(
                    ee.Geometry.Polygon(
                        geostore.get('geojson').get('features')[n].get('geometry').get('coordinates')[i]
                    )
                )
            )
            
        # Create a FeatureCollection from the list.
        feature_collections.append(ee.FeatureCollection(features))
    return feature_collections

def check_status_data(task, file_paths):
    status_list = list(map(lambda x: str(x), task.list()[:len(file_paths)])) 
    status_list = list(map(lambda x: x[x.find("(")+1:x.find(")")], status_list))
    
    return status_list

def removekey(dictionary, key):
    if key in dictionary.keys():
        del dictionary[key]
    return dictionary
