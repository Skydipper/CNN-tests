import ee
import Skydipper
import sqlalchemy
import folium
import json
import time
import pandas as pd
import requests
import getpass
import argparse
from argparse import Namespace
from ee.cli.utils import CommandLineConfig
from ee.cli.commands import PrepareModelCommand
from shapely.geometry import shape
from google.cloud import storage
from google.cloud.storage import blob
from googleapiclient import discovery
from googleapiclient import errors
from oauth2client.client import GoogleCredentials
from tensorflow.python.tools import saved_model_utils

from .utils import df_from_query, df_to_db, df_to_csv, polygons_to_geoStoreMultiPoligon, get_geojson_string,\
    min_max_values, normalize_ee_images, get_image_ids, GeoJSONs_to_FeatureCollections, check_status_data,\
    removekey
import ee_collection_specifics

class Trainer(object):
    """
    Training of Deep Learning models in Skydipper
    ----------
    privatekey_path: string
        A string specifying the direction of a json keyfile on your local filesystem
        e.g. "/Users/me/.privateKeys/key_with_bucket_permissions.json"
    """
    def __init__(self, privatekey_path):
        self.privatekey_path = privatekey_path
        self.storage_client = storage.Client.from_service_account_json(self.privatekey_path)
        self.db_url = 'postgresql://postgres:postgres@0.0.0.0:5432/geomodels'
        self.engine = sqlalchemy.create_engine(self.db_url)
        self.slugs_list = ["Sentinel-2-Top-of-Atmosphere-Reflectance",
              "Landsat-7-Surface-Reflectance",
              "Landsat-8-Surface-Reflectance",
              "USDA-NASS-Cropland-Data-Layers",
              "USGS-National-Land-Cover-Database",
              "Lake-Water-Quality-100m"]
        self.table_names = self.engine.table_names()
        #self.datasets_api = Skydipper.Collection(search=' '.join(self.slugs_list), object_type=['dataset'], app=['skydipper'], limit=len(self.slugs_list))
        self.ee_tiles = 'https://earthengine.googleapis.com/map/{mapid}/{{z}}/{{x}}/{{y}}?token={token}' 
        self.datasets = df_from_query(self.engine, 'dataset') 
        self.images = df_from_query(self.engine, 'image')  
        self.models = df_from_query(self.engine, 'model') 
        self.versions = df_from_query(self.engine, 'model_versions') 
        self.bucket = 'geo-ai'
        self.project_id = 'skydipper-196010'
        # Get a Python representation of the AI Platform Training services
        self.credentials = GoogleCredentials.from_stream(self.privatekey_path)
        self.ml = discovery.build('ml', 'v1', credentials = self.credentials)
        ee.Initialize()

        # TODO
        #self.datasets_api = Skydipper.Collection(search=' '.join(self.slugs_list), object_type=['dataset'], app=['skydipper'], limit=len(self.slugs_list))

    def get_token(self, email):
        password = getpass.getpass('Skydipper login password:')

        payload = {
            "email": f"{email}",
            "password": f"{password}"
        }

        url = f'https://api.skydipper.com/auth/login'

        headers = {'Content-Type': 'application/json'}

        r = requests.post(url, data=json.dumps(payload), headers=headers)

        self.token = r.json().get('data').get('token')

    def composite(self, slugs=["Sentinel-2-Top-of-Atmosphere-Reflectance"], init_date='2019-01-01', end_date='2019-12-31', lat=39.31, lon=0.302, zoom=6):
        """
        Returns a folium map with the composites.
        Parameters
        ----------
        slugs: list
            A list of dataset slugs to display on the map.
        init_date: string
            Initial date of the composite.
        end_date: string
            Last date of the composite.
        lat: float
            A latitude to focus the map on.
        lon: float
            A longitude to focus the map on.
        zoom: int
            A z-level for the map.
        """
        self.slugs = slugs
        self.init_date = init_date
        self.end_date= end_date

        map = folium.Map(
                location=[lat, lon],
                zoom_start=zoom,
                tiles='OpenStreetMap',
                detect_retina=True,
                prefer_canvas=True
        )

        composites = []
        for n, slug in enumerate(slugs):
            composites.append(ee_collection_specifics.Composite(slug)(init_date, end_date))

            mapid = composites[n].getMapId(ee_collection_specifics.vizz_params_rgb(slug))
            tiles_url = self.ee_tiles.format(**mapid)
            folium.TileLayer(
            tiles=tiles_url,
            attr='Google Earth Engine',
            overlay=True,
            name=str(ee_collection_specifics.ee_bands_rgb(slug))).add_to(map)

        self.composites = composites

        map.add_child(folium.LayerControl())
        return map

    def create_geostore_from_geojson(self, attributes, zoom=6):
        """Parse valid geojson into a geostore object and register it to a
        Gestore object on a server. 
        Parameters
        ----------
        attributes: list
            List of geojsons with the trainig, validation, and testing polygons.
        zoom: int
            A z-level for the map.
        """
        # Get MultiPolygon geostore object
        self.geostore = polygons_to_geoStoreMultiPoligon(attributes)

        nFeatures = len(self.geostore.get('geojson').get('features'))

        nPolygons = {}
        for n in range(nFeatures):
            multipoly_type = self.geostore.get('geojson').get('features')[n].get('properties').get('name')
            nPolygons[multipoly_type] = len(self.geostore.get('geojson').get('features')[n].get('geometry').get('coordinates'))
    
        for multipoly_type in nPolygons.keys():
            print(f'Number of {multipoly_type} polygons:', nPolygons[multipoly_type])

        self.nPolygons = nPolygons
        
        # TODO(replace code with SkyPy)
        #self.multipolygon = Skydipper.Geometry(attributes=geostore) # This is here commented until SkyPy works again
        #self.geostore_id = multipolygon.id # This is here commented until SkyPy works again
        # Register geostore object on a server. Return the object, and instantiate a Geometry.
        if self.token:
            header= {
                'Authorization': 'Bearer ' + self.token,
                'Content-Type':'application/json'
                    }
            url = 'https://api.skydipper.com/v1/geostore'
            r = requests.post(url, headers=header, json=self.geostore)

            self.multipolygon = r.json().get('data').get('attributes')
            self.geostore_id = r.json().get('data').get('id')

        else:
            raise ValueError(f'Token is required use get_token() method first.')

        # Returns a folium map with the polygons
        #features = self.multipolygon.attributes['geojson']['features']
        features = self.geostore['geojson']['features']
        if len(features) > 0:
            shapely_geometry = [shape(feature['geometry']) for feature in features]
        else:
            shapely_geometry = None
    
        self.centroid = list(shapely_geometry[0].centroid.coords)[0][::-1]
    
        #bbox = self.multipolygon.attributes['bbox']
        #self.bounds = [bbox[2:][::-1], bbox[:2][::-1]]        

        map = folium.Map(location=self.centroid, zoom_start=zoom)
        #map.fit_bounds(self.bounds)

        if hasattr(self, 'composites'):
            for n, slug in enumerate(self.slugs):
                mapid = self.composites[n].getMapId(ee_collection_specifics.vizz_params_rgb(slug))
                tiles_url = self.ee_tiles.format(**mapid)
                folium.TileLayer(
                tiles=tiles_url,
                attr='Google Earth Engine',
                overlay=True,
                name=str(ee_collection_specifics.ee_bands_rgb(slug))).add_to(map)

        nFeatures = len(features)
        colors = ['#64D1B8', 'red', 'blue']
        for n in range(nFeatures):
            style_function = lambda x: {
                'fillOpacity': 0.0,
                    'weight': 4,
                    'color': colors[0]
                    }
            folium.GeoJson(data=get_geojson_string(features[n]['geometry']), style_function=style_function,\
                 name=features[n].get('properties').get('name')).add_to(map)
        
        map.add_child(folium.LayerControl())
        return map

    def normalize_images(self, scale, slugs=None, init_date=None, end_date=None, norm_type='global', zoom=6):
        """
        Returns the min/max values of each band in a composite.
        Parameters
        ----------
        scale: float
            Scale of the images.
        slugs: list
            A list of dataset slugs to display on the map.
        init_date: string
            Initial date of the composite.
        end_date: string
            Last date of the composite.
        norm_type: string
            Normalization type. Posible values:
             - 'global': normalization over the whole globe.
             - 'geostore': normalization over th polygond inside the geostore
        zoom: int
            A z-level for the map.
        """
        self.scale = scale
        self.norm_type = norm_type
        if hasattr(self, 'composites'):
            self.slugs = self.slugs
            self.init_date = self.init_date
            self.end_date = self.end_date
        else:
            if (slugs != None) and (init_date != None) and (end_date != None):
                self.slugs = slugs
                self.init_date = init_date
                self.end_date = end_date
            else:
                raise ValueError(f"Missing 3 required positional arguments: 'slugs', 'init_date', and 'end_date'")

        # Get normalization values
        self.values = []
        for slug in self.slugs:
            # Create composite
            image = ee_collection_specifics.Composite(slug)(self.init_date, self.end_date)

            bands = ee_collection_specifics.ee_bands(slug)
            image = image.select(bands)

            if ee_collection_specifics.normalize(slug):
                # Get min/man values for each band
                if (self.norm_type == 'geostore'):
                    if hasattr(self, 'geostore'):
                        value = min_max_values(image, slug, self.scale, norm_type=self.norm_type, geostore=self.geostore)
                    else:
                        raise ValueError(f"Missing geostore attribute. Please run create_geostore_from_geojson() first")
                else:
                    value = min_max_values(image, slug, self.scale, norm_type=self.norm_type)
            else:
                value = {}
            self.values.append(json.dumps(value))
        
        # Populate image table
        for n, slug in enumerate(self.slugs):
            dataset_id = self.datasets[self.datasets['slug'] == slug].index[0]

            # Populate image table
            if self.norm_type == 'geostore':
                condition = self.images[['dataset_id', 'scale', 'init_date', 'end_date', 'norm_type', 'geostore_id']]\
                                .isin([dataset_id, self.scale, self.init_date, self.end_date, self.norm_type, self.geostore_id]).all(axis=1).any()
                dictionary = dict(zip(list(self.images.keys()), [[dataset_id], [''], [self.scale], [self.init_date], [self.end_date], [self.values[n]], [self.norm_type], [self.geostore_id]]))
            else:
                condition = self.images[['dataset_id', 'scale', 'init_date', 'end_date', 'norm_type']]\
                                .isin([dataset_id, self.scale, self.init_date, self.end_date, self.norm_type]).all(axis=1).any()
                dictionary = dict(zip(list(self.images.keys()), [[dataset_id], [''], [self.scale], [self.init_date], [self.end_date], [self.values[n]], [self.norm_type], ['']]))

            if not condition:
                # Append values to table
                self.images = self.images.append(pd.DataFrame(dictionary), ignore_index = True, sort=False)

        # Returns a folium map with normalized images
        map = folium.Map(location=self.centroid, zoom_start=6)
        #map.fit_bounds(self.bounds)

        self.norm_composites = []
        for n, slug in enumerate(self.slugs):
        
            dataset_id = self.datasets[self.datasets['slug'] == slug].index[0]
            value = json.loads(self.values[n])

            # Create composite
            image = ee_collection_specifics.Composite(slug)(self.init_date, self.end_date)

            # Normalize images
            if bool(value): 
                image = normalize_ee_images(image, slug, value)

            for params in ee_collection_specifics.vizz_params(slug):
                mapid = image.getMapId(params)
                folium.TileLayer(
                tiles=self.ee_tiles.format(**mapid),
                attr='Google Earth Engine',
                overlay=True,
                name=str(params['bands']),
              ).add_to(map)

            self.norm_composites.append(image)

        map.add_child(folium.LayerControl())

        return map

    def select_bands(self, input_bands, output_bands):
        """
        Selects input and output bands.
        Parameters
        ----------
        input_bands: list
            List of input bands.
        output_bands: list
            List of output bands.
        """
        self.bands = [input_bands, output_bands]

        # Populate image table
        for n, slug in enumerate(self.slugs):
        
            dataset_id = self.datasets[self.datasets['slug'] == slug].index[0]

            if self.norm_type == 'geostore':
                df = self.images[(self.images['dataset_id'] == dataset_id) & 
                            (self.images['scale'] == self.scale) & 
                            (self.images['init_date'] == self.init_date) & 
                            (self.images['end_date'] == self.end_date) & 
                            (self.images['norm_type'] == self.norm_type) & 
                            (self.images['geostore_id'] == self.geostore_id)
                           ].copy()
            else:
                df = self.images[(self.images['dataset_id'] == dataset_id) & 
                            (self.images['scale'] == self.scale) & 
                            (self.images['init_date'] == self.init_date) & 
                            (self.images['end_date'] == self.end_date) & 
                            (self.images['norm_type'] == self.norm_type)
                           ].copy()

            # Take rows where bands_selections column is empty
            df1 = df[df['bands_selections'] == ''].copy()

            if df1.any().any():
                # Take first index
                index = df1.index[0]
                self.images.at[index, 'bands_selections'] = str(self.bands[n])
            else:
                if not self.images[['dataset_id', 'bands_selections', 'scale', 'init_date', 'end_date', 'norm_type']].isin(
                    [dataset_id, str(self.bands[n]), self.scale, self.init_date, self.end_date, self.norm_type]).all(axis=1).any():

                    df2 = df.iloc[0:1].copy()
                    df2.at[df2.index[0], 'bands_selections'] = str(self.bands[n])
                    self.images = self.images.append(df2, ignore_index = True)

    def stack_images(self, feature_collections):
        """
        Stack the 2D images (input and output images of the Neural Network) 
        to create a single image from which samples can be taken
        """
        for n, slug in enumerate(self.slugs):

            dataset_id = self.datasets[self.datasets['slug'] == slug].index[0]

            df = self.images[(self.images['dataset_id'] == dataset_id) & 
                        (self.images['bands_selections'] == str(self.bands[n])) & 
                        (self.images['scale'] == self.scale) & 
                        (self.images['init_date'] == self.init_date) & 
                        (self.images['end_date'] == self.end_date) &
                        (self.images['norm_type'] == self.norm_type)
                    ].copy()

            values = json.loads(df['bands_min_max'].iloc[0])

            # Stack normalized composites
            if n == 0:
                image_stack = self.norm_composites[n].select(self.bands[n])
            else:
                image_stack = ee.Image.cat([image_stack,self.norm_composites[n].select(self.bands[n])]).float()

        if self.kernel_size == 1:
            self.base_names = ['training_pixels', 'validation_pixels', 'test_pixels']
            # Sample pixels
            vector = image_stack.sample(region = feature_collections[0], scale = self.scale,\
                                        numPixels=self.sample_size, tileScale=4, seed=999)

            # Add random column
            vector = vector.randomColumn(seed=999)

            # Partition the sample approximately 60%, 20%, 20%.
            self.training_dataset = vector.filter(ee.Filter.lt('random', 0.6))
            self.validation_dataset = vector.filter(ee.Filter.And(ee.Filter.gte('random', 0.6),\
                                                            ee.Filter.lt('random', 0.8)))
            self.test_dataset = vector.filter(ee.Filter.gte('random', 0.8))

            # Training and validation size
            self.training_size = self.training_dataset.size().getInfo()
            self.validation_size = self.validation_dataset.size().getInfo()
            self.test_size = self.test_dataset.size().getInfo()

        if self.kernel_size > 1:
            self.base_names = ['training_patches', 'validation_patches', 'test_patches']
            # Convert the image into an array image in which each pixel stores (kernel_size x kernel_size) patches of pixels for each band.
            list = ee.List.repeat(1, self.kernel_size)
            lists = ee.List.repeat(list, self.kernel_size)
            kernel = ee.Kernel.fixed(self.kernel_size, self.kernel_size, lists)

            arrays = image_stack.neighborhoodToArray(kernel)

            # Training and validation size
            nFeatures = len(self.geostore.get('geojson').get('features'))
            nPolygons = {}
            for n in range(nFeatures):
                multipoly_type = self.geostore.get('geojson').get('features')[n].get('properties').get('name')
                nPolygons[multipoly_type] = len(self.geostore.get('geojson').get('features')[n].get('geometry').get('coordinates'))

            self.training_size = nPolygons['training']*self.sample_size
            self.validation_size = nPolygons['validation']*self.sample_size
            self.test_size = nPolygons['test']*self.sample_size


    def start_TFRecords_task(self, feature_collections, feature_lists):
        """
        Create TFRecord's exportation task
        """
        # Folder path to save the data
        folder = 'Data/'+str(self.image_ids[0])+'_'+ str(self.image_ids[1])+'/'+str(self.geostore_id)+'/'+str(self.kernel_size)+'/'+str(self.sample_size)

        # These numbers determined experimentally.
        nShards  = int(self.sample_size/20) # Number of shards in each polygon.

        if self.kernel_size == 1:
            # Export all the training validation and test data.   
            self.file_paths = []
            for n, dataset in enumerate([self.training_dataset, self.validation_dataset, self.test_dataset]):

                self.file_paths.append(self.bucket+ '/' + folder + '/' + self.base_names[n])

                # Create the tasks.
                task = ee.batch.Export.table.toCloudStorage(
                  collection = dataset,
                  description = 'Export '+self.base_names[n],
                  fileNamePrefix = folder + '/' + self.base_names[n],
                  bucket = self.bucket,
                  fileFormat = 'TFRecord',
                  selectors = self.bands[0] + self.bands[1])

                task.start()

        if self.kernel_size > 1:
             # Export all the training validation and test data. (in many pieces), with one task per geometry.     
            self.file_paths = []
            for i, feature in enumerate(feature_collections):
                for g in range(feature.size().getInfo()):
                    geomSample = ee.FeatureCollection([])
                    for j in range(nShards):
                        sample = arrays.sample(
                            region = ee.Feature(feature_lists[i].get(g)).geometry(), 
                            scale = self.scale, 
                            numPixels = self.sample_size / nShards, # Size of the shard.
                            seed = j,
                            tileScale = 8
                        )
                        geomSample = geomSample.merge(sample)

                    desc = self.base_names[i] + '_g' + str(g)

                    self.file_paths.append(self.bucket+ '/' + folder + '/' + desc)

                    task = ee.batch.Export.table.toCloudStorage(
                        collection = geomSample,
                        description = desc, 
                        bucket = self.bucket, 
                        fileNamePrefix = folder + '/' + desc,
                        fileFormat = 'TFRecord',
                        selectors = self.bands[0] + self.bands[1]
                    )
                    task.start()

        return task

    def export_TFRecords(self, sample_size, kernel_size):
        """
        Export TFRecords to GCS.
        Parameters
        ----------
        sample_size: int
            Number of samples to extract from each polygon.
        kernel_size: int
            An integer specifying the height and width of the 2D images.
        """
        self.sample_size = sample_size
        self.kernel_size = kernel_size
        # Get image ids
        self.image_ids = get_image_ids(self.images, self.datasets, self.slugs,\
                                       self.bands, self.scale, self.init_date, self.end_date, self.norm_type)
        # Convert the GeoJSON to feature collections
        feature_collections = GeoJSONs_to_FeatureCollections(self.geostore)
        
        # Convert the feature collections to lists for iteration.
        feature_lists = list(map(lambda x: x.toList(x.size()), feature_collections))

        # Stack the 2D images to create a single image from which samples can be taken
        self.stack_images(feature_collections)

        df = self.versions[['input_image_id', 'output_image_id', 'geostore_id', 'kernel_size', 'sample_size']\
                          ].isin([self.image_ids[0], self.image_ids[1], self.geostore_id, self.kernel_size, self.sample_size]).copy()
        if not df.all(axis=1).any():
            task = self.start_TFRecords_task(feature_collections, feature_lists)
        elif not (self.versions[df.all(axis=1)]['data_status'] == 'COMPLETED').all():
            task = self.start_TFRecords_task(feature_collections, feature_lists)

        # Populate model_versions tables
        if (self.versions.empty) or not df.all(axis=1).any():
            dictionary = dict(zip(list(self.versions.keys()), [[-9999], [''], [self.image_ids[0]], [self.image_ids[1]], [self.geostore_id], [self.kernel_size], [self.sample_size], [json.dumps({})], [-9999], [''], [''], [False], [False]]))
            self.versions = self.versions.append(pd.DataFrame(dictionary), ignore_index = True, sort=False)

        # Save task status
        if (df.empty) or not (self.versions[df.all(axis=1)]['data_status'] == 'COMPLETED').all():
            print('Exporting TFRecords to GCS:')
            status_list = check_status_data(task, self.file_paths)
            index = self.versions.index[-1]
            while not status_list == ['COMPLETED'] * len(self.file_paths):
                status_list = check_status_data(task, self.file_paths)

                #Save temporal status in table
                tmp_status = json.dumps(dict(zip(self.file_paths, status_list)))
                self.versions.at[index, 'data_status'] = tmp_status
                print('Temporal status: ', tmp_status)

                time.sleep(60)

            # Save final status in table
            self.versions.at[index, 'data_status'] = "COMPLETED"  
            print('Final status: COMPLETED')
            
            # TODO
            # Save image and model_versions tables
            df_to_csv(self.images, "image")
            df_to_csv(self.versions, "model_versions")
            df_to_db(self.images, self.engine, "image")
            df_to_db(self.versions, self.engine, "model_versions")

    def check_trainig_status(self, ml, project, job_name):
        # Monitoring the training job
        jobId = '{}/jobs/{}'.format(project, job_name)
        request = ml.projects().jobs().get(name=jobId)
        # Make the call.
        try:
            response = request.execute()
        except errors.HttpError as err:
            # Something went wrong, print out some information.
            print('There was an error monitoring the training job. Check the details:')
            print(err._get_reason())

        return response['state']
               
    def train_model_ai_platform(self, model_type='CNN', model_output='segmentation', model_architecture='segnet',\
                                model_name = None, model_description='', output_activation='', batch_size=32,\
                                epochs=25, shuffle_size=2000, learning_rate=1e-3,\
                                loss=None,  metrics=None):
        """
        Trains the model in AI Platform.
        Parameters
        ----------
        model_type: string
            Type of neural network. We support: 
                - Convolutional Neural Network (CNN)
                - multilayer perceptron (MLP)
        model_output: string
            Output of the neural network. We support:
                - regression
                - segmentation
        model_architecture: string
            Name of the architecture to be used (e.g.: segnet, deepvel, unet ...)
        model_name: string
            Name of the model
        model_description: string
            Description of the model
        output_activation: string
            Name of the last activation function. We support all the activations from https://keras.io/activations/
        batch_size: int
            A number of samples processed before the model is updated. 
            The size of a batch must be more than or equal to one and less than or equal to the number of samples in the training dataset.
        epochs: int
            Number of complete passes through the training dataset.
        shuffle_size: int
            Number of samples to be shuffled.
        learning_rate: float
            A hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.
        loss: string
            Name of a method to evaluate how well the model fits the given data. We support all the loos functions from https://keras.io/losses/
        metrics: list of strings
            Name of a function that is used to judge the performance of your model. We support all the metric functions from https://keras.io/metrics/
        """
        self.model_types = ['CNN', 'MLP'] 
        self.CNN_outputs = ['regression', 'segmentation']  
        self.CNN_architectures = ['deepvel','segnet','unet']
        self.MLP_outputs = ['regression']  
        self.MLP_architectures = ['sequential1']

        self.model_structure = {'model_type': {
            'CNN': {'model_output': {
                'regression': {'model_architecture': [
                        'deepvel','segnet','unet']},
                'segmentation': {'model_architecture': [
                        'deepvel','segnet','unet']}}
            },
            'MLP': {'model_output': {
                'regression': {'model_architecture': [
                        'sequential1']}}}}}

        self.region = 'us-central1'
        self.main_trainer_module = 'trainer.task'

        error_dic = {
            'CNN': {
                'outputs': self.CNN_outputs,
                'architectures': self.CNN_architectures,
                'kernel_size': (self.kernel_size > 1),
                'kernel_error': 'kernel_size > 1'
            },
            'MLP': {
                'outputs': self.MLP_outputs,
                'architectures': self.MLP_architectures,
                'kernel_size': (self.kernel_size == 1),
                'kernel_error': 'kernel_size = 1'
            }
        }

        if model_type in self.model_types:
            if (model_output in error_dic[model_type]['outputs']) and (model_architecture in error_dic[model_type]['architectures']):
                self.model_output = model_output
                self.model_architecture = model_architecture
                if error_dic[model_type]['kernel_size']:
                    self.model_type = model_type
                else:
                    m = error_dic[model_type]['kernel_error']
                    raise ValueError(f'Model type {model_type} only supported when {m}. Current kernel_size is equal to {str(self.kernel_size)}')
            else:
                raise ValueError(f'Unsupported model structure. Check compatibilities: {json.dumps(self.model_structure)}')
        else:
            raise ValueError(f'Unsupported model type. Choose between [CNN, MLP]')

        df = self.models[['model_type', 'model_output', 'output_image_id']].isin([self.model_type, self.model_output, self.image_ids[1]]).copy()
        if df.all(axis=1).any():
            self.model_name = self.models[df.all(axis=1)]['model_name'].iloc[0]
            self.model_description = self.models[df.all(axis=1)]['model_description'].iloc[0]
            print(f'Model already exists with name: {self.model_name}.')
            print(f'And description {self.model_description}.')
        elif model_name:
            # check if model_name already exists
            if not model_name in list(self.models['model_name']):
                self.model_name = model_name
                self.model_description = model_description
                print(f'Model name assigned: {self.model_name} with description: {self.model_description}')
            else:
                raise ValueError(f'{model_name} already exists. Please change model_name attribute.')
        else:
            raise ValueError(f'model_name attribute is required.')

        if not loss:
            if self.model_output == 'regression':
                self.loss = 'mse'
            if self.model_output == 'segmentation':
                self.loss = 'accuaracy'
        if not metrics:
            if self.model_output == 'regression':
                self.metrics = ['mse']
            if self.metrics == 'segmentation':
                self.loss = ['accuaracy']

        # Training parameters
        self.training_params = {
            "bucket": self.bucket,
            "base_names": self.base_names,
            "data_dir": 'gs://' + self.bucket + '/Data/' + str(self.image_ids[0])+'_'+ str(self.image_ids[1])+'/'+str(self.geostore_id)+'/'+str(self.kernel_size)+'/'+str(self.sample_size),
            "in_bands": self.bands[0],
            "out_bands": self.bands[1],
            "kernel_size": int(self.kernel_size),
            "training_size": self.training_size,
            "validation_size": self.validation_size,
            "model_type": self.model_type,
            "model_output": self.model_output,
            "model_architecture": self.model_architecture,
            "output_activation": output_activation,
            "batch_size": batch_size,
            "epochs": epochs,
            "shuffle_size": shuffle_size,
            "learning_rate": learning_rate,
            "loss": self.loss,
            "metrics": self.metrics
        }

        # Populate model table
        if not df.all(axis=1).any():
            dictionary = dict(zip(list(self.models.keys()), [[self.model_name], [self.model_type], [self.model_output], [self.model_description], [self.image_ids[1]]]))
            self.models = self.models.append(pd.DataFrame(dictionary), ignore_index = True, sort=False)

        self.model_id = self.models[(self.models['model_type'] == self.model_type) & (self.models['model_output'] == self.model_output) & (self.models['output_image_id'] == self.image_ids[1])].index[0]

        # Populate model_versions table
        df = self.versions[['input_image_id', 'output_image_id', 'geostore_id', 'kernel_size', 'sample_size']\
                          ].isin([self.image_ids[0], self.image_ids[1], self.geostore_id, self.kernel_size, self.sample_size]).copy()
        df = self.versions.copy()
        df['training_params'] = df['training_params'].apply(lambda x : removekey(json.loads(x),['job_dir', 'training_size', 'validation_size']))

        # Check if the version already exists
        if (df['training_params'] == removekey(self.training_params.copy(), ['training_size', 'validation_size'])).any():
            # Get version id
            self.version_id = df[df['training_params'] == removekey(self.training_params.copy(), ['training_size', 'validation_size'])].index[0]
            # Check status
            status = df.iloc[self.version_id]['training_status']
            print('Version already exists with training status equal to:', status)

            # Get training version
            self.training_version = df.iloc[self.version_id]['version']

            # Add job directory
            self.training_params['job_dir'] = 'gs://' + self.bucket + '/Models/' + str(self.model_id) + '/' +  str(self.training_version) + '/'

            if status == 'SUCCEEDED':
                print('The training job successfully completed.')
                return
            if (status == 'CANCELLED') or (status == 'FAILED'):
                print(f'The training job was {status}.')
                if status == 'CANCELLED':  
                    print('Start training again.')
                if status == 'FAILED': 
                    print('Change training parameters and try again.')
                # Update job name
                job_name = 'job_v' + str(int(time.time()))

                # Save training version and clear status
                self.versions.at[self.version_id, 'training_params'] =  json.dumps(self.training_params)
                self.versions.at[self.version_id, 'training_status'] = ''

                # Remove job_dir
                tmp_path = self.training_params['job_dir'].replace(f'gs://{self.bucket}/','')
                bucket = self.storage_client.get_bucket(self.bucket)
                blob = bucket.blob(tmp_path)

                blob.delete()

        # Create new version  
        else:
            print('Create new version')
            # New training version and job name
            self.training_version = str(int(time.time()))
            job_name = 'job_v' + self.training_version

            # Add job directory
            self.training_params['job_dir'] = 'gs://' + self.bucket + '/Models/' + str(self.model_id) + '/' +  str(self.training_version) + '/'

            df = self.versions[['input_image_id', 'output_image_id', 'geostore_id', 'kernel_size', 'sample_size', 'data_status']].isin(
                [self.image_ids[0], self.image_ids[1], self.geostore_id, self.kernel_size, self.sample_size, 'COMPLETED']).copy()

            # Check if untrained version already exists
            if (df.all(axis=1).any()):
                self.version_id = df[df.all(axis=1)].index[0]

                self.versions.at[self.version_id, 'model_id'] = self.model_id
                self.versions.at[self.version_id, 'model_architecture'] = self.model_architecture
                self.versions.at[self.version_id, 'training_params'] = json.dumps(self.training_params)
                self.versions.at[self.version_id, 'version'] = self.training_version

            else:
                dictionary = dict(zip(list(self.versions.keys()), [[''], [''], [self.image_ids[0]], [self.image_ids[1]], [self.geostore_id], [self.kernel_size], [self.sample_size], [''], [''], ['COMPLETED'], [''], [''], ['']]))
                self.versions = self.versions.append(pd.DataFrame(dictionary), ignore_index = True, sort=False)
                self.version_id = self.versions.index[-1]

                self.versions.at[self.version_id, 'model_id'] = int(self.model_id)
                self.versions.at[self.version_id, 'model_architecture'] = self.model_architecture
                self.versions.at[self.version_id, 'training_params'] = json.dumps(self.training_params)
                self.versions.at[self.version_id, 'version'] = int(self.training_version)


        # set version table's types
        self.versions = self.versions.astype({'model_id': 'int64', 
                                    'version': 'int64', 
                                    'eeified': bool, 
                                    'deployed': bool})

        # Save training parameters
        params_path = 'Models/' + str(self.model_id) + '/' +  str(self.training_version) + '/training_params.json'
        bucket = self.storage_client.get_bucket(self.bucket)
        blob = bucket.blob(params_path)

        blob.upload_from_string(
            data=json.dumps(self.training_params),
            content_type='application/json',
            client=self.storage_client
        )

        # Submit a training job to AI Platform
        self.training_inputs = {'scaleTier': 'CUSTOM',             
            'masterType': 'large_model_v100', # A single NVIDIA Tesla V100 GPU 
            'packageUris': [f'gs://{self.bucket}/Train/trainer-0.2.tar.gz'],
            'pythonModule': self.main_trainer_module,
            'args': ['--params-file', params_path],
            'region': self.region,
            'jobDir': self.training_params['job_dir'],
            'runtimeVersion': '1.14',
            'pythonVersion': '3.5'}

        job_spec = {'jobId': job_name, 'trainingInput': self.training_inputs}

        print('Creating training job: ' + job_name)
        # Save your project ID in the format the APIs need
        project = 'projects/{}'.format(self.project_id)

        # Create a request to call projects.jobs.create.
        request = self.ml.projects().jobs().create(body=job_spec,
                      parent=project)

        # Make the call.
        try:
            response = request.execute()
            print(response)

        except errors.HttpError as err:
            # Something went wrong, print out some information.
            print('There was an error creating the training job. Check the details:')
            print(err._get_reason())
            
        # Monitoring the training job
        self.status = self.check_trainig_status(self.ml, project, job_name)
        while not self.status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            self.status = self.check_trainig_status(self.ml, project, job_name)
            #Save temporal status in table
            self.versions.at[self.version_id, 'training_status'] = self.status    
            print('Current training status: ' +  self.status)

            time.sleep(60)

        #Save final status in table
        self.versions.at[self.version_id, 'training_status'] = self.status

        # TODO
        # Save image and model_versions tables
        df_to_csv(self.models, "model")
        df_to_csv(self.versions, "model_versions")
        df_to_db(self.models, self.engine, "model")
        df_to_db(self.versions, self.engine, "model_versions")

    def check_deployment_status(self, ml, project, model_name, version_name):
        # Monitoring the training job
        versionId = f'{project}/models/{model_name}/versions/{version_name}'
        request = ml.projects().models().versions().get(name=versionId)
        # Make the call.
        try:
            response = request.execute()
        except errors.HttpError as err:
            # Something went wrong, print out some information.
            print('There was an error monitoring the version creation. Check the details:')
            print(err._get_reason())

        return response['state']

    def deploy_model_ai_platform(self, EEify=True):
        """
        Deploy trained model to AI Platform.
        Parameters
        ----------
        EEify: boolean
            Before we can use the model in Earth Engine, it needs to be hosted by AI Platform. 
            But before we can host the model on AI Platform we need to EEify (a new word!) it. 
            The EEification process merely appends some extra operations to the input and outputs 
            of the model in order to accomdate the interchange format between pixels from Earth Engine (float32) 
            and inputs to AI Platform (base64).
        """
        if EEify:
            print('Preparing the model for making predictions in Earth Engine')
            model_path = self.training_params.get('job_dir') + 'model/'

            meta_graph_def = saved_model_utils.get_meta_graph_def(model_path, 'serve')
            inputs = meta_graph_def.signature_def['serving_default'].inputs
            outputs = meta_graph_def.signature_def['serving_default'].outputs

            # Just get the first thing(s) from the serving signature def.  i.e. this
            # model only has a single input and a single output.
            input_name = None
            for k,v in inputs.items():
                input_name = v.name
                break
            
            output_name = None
            for k,v in outputs.items():
                output_name = v.name
                break
            
            # Make a dictionary that maps Earth Engine outputs and inputs to 
            # AI Platform inputs and outputs, respectively.
            input_dict = json.dumps({input_name: "array"}) #"'" + json.dumps({input_name: "array"}) + "'"
            output_dict = json.dumps({output_name: "prediction"}) #"'" + json.dumps({output_name: "prediction"}) + "'"

            # Put the EEified model next to the trained model directory.
            EEified_path = self.training_params.get('job_dir') + 'eeified/'

            # Send EEified model to GCS
            parser = argparse.ArgumentParser()
            args = Namespace(source_dir=model_path,
                             dest_dir=EEified_path,
                             input=input_dict,
                             output=output_dict,
                             tag=None,
                             variables=None
                            )
            config = CommandLineConfig()

            eeify_model = PrepareModelCommand(parser)

            eeify_model.run(args, config)

            # Populate model_versions table
            self.versions.at[self.version_id, 'eeified'] = True

        # Deployed the model to AI Platform
        version_name = 'v' + str(self.training_version)
        print(f'Deploying {version_name} version of {self.model_name} model to AI Platform')
        
        # Get deployed model list
        # Create a request to call projects().models().list.
        project = 'projects/{}'.format(self.project_id)
        request = self.ml.projects().models().list(parent=project)

        # Make the call.
        try:
            response = request.execute()
        except errors.HttpError as err:
            # Something went wrong, print out some information.
            print('There was an error creating the training job. Check the details:')
            print(err._get_reason())

        models_list = list(map(lambda x: x.get('name'), response.get('models')))

        # Create model in AI platform
        if not f'projects/{self.project_id}/models/{self.model_name}' in models_list:
            # Create model model in AI platform
            # Create a dictionary with the fields from the request body.
            request_dict = {'name': self.model_name,
                           'description': self.model_description}
            # Create a request to call projects.models.create.
            request = self.ml.projects().models().create(
                          parent=project, body=request_dict)

            # Make the call.
            try:
                response = request.execute()
                print(response)
            except errors.HttpError as err:
                # Something went wrong, print out some information.
                print('There was an error creating the model. Check the details:')
                print(err._get_reason())

        # Create model version
        # Create a dictionary with the fields from the request body.
        request_dict = {
            'name': version_name,
            'deploymentUri': EEified_path,
            'runtimeVersion': '1.14',
            'pythonVersion': '3.5',
            'framework': 'TENSORFLOW',
            'autoScaling': {
                "minNodes": 10
            }
        }

        # Create a request to call projects.models.versions.create.
        request = self.ml.projects().models().versions().create(
            parent=f'projects/{self.project_id}/models/{self.model_name}',
            body=request_dict
        )

        # Make the call.
        try:
            response = request.execute()
            print(response)
        except errors.HttpError as err:
            # Something went wrong, print out some information.
            print('There was an error creating the model. Check the details:')
            print(err._get_reason())

        # Save deployment status
        self.status = self.check_deployment_status(self.ml, project, self.model_name, version_name)
        while not self.status in 'READY':
            self.status = self.check_deployment_status(self.ml, project, self.model_name, version_name)
            #Save temporal status in table
            self.versions.at[self.version_id, 'deployed'] = False    
            print('Current training status: ' +  self.status)

            time.sleep(60)

        #Save final status in table
        self.versions.at[self.version_id, 'deployed'] = True

        # TODO
        # Save image and model_versions tables
        df_to_csv(self.versions, "model_versions")
        df_to_db(self.versions, self.engine, "model_versions")

class Validator(object):
    """
    Validation of Deep Learning models in Skydipper
    """
    def __init__(self):
        self.db_url = 'postgresql://postgres:postgres@0.0.0.0:5432/geomodels'
        self.engine = sqlalchemy.create_engine(self.db_url)
        self.table_names = self.engine.table_names()
        #self.datasets_api = Skydipper.Collection(search=' '.join(self.slugs_list), object_type=['dataset'], app=['skydipper'], limit=len(self.slugs_list))
        self.datasets = df_from_query(self.engine, 'dataset') 
        self.images = df_from_query(self.engine, 'image')  
        self.models = df_from_query(self.engine, 'model') 
        self.versions = df_from_query(self.engine, 'model_versions') 
        self.bucket = 'geo-ai'
        self.project_id = 'skydipper-196010'

    def select_model(self, model_name):
        """
        Selects model.
        Parameters
        ----------
        model_name: string
            Model name.
        """
        self.model_name = model_name
        self.model_id = self.models[self.models['model_name'] == model_name].index[0]
        self.model_type = self.models.iloc[self.model_id]['model_type']
        self.model_output = self.models.iloc[self.model_id]['model_output']
        self.version_names = list(map(lambda x: int(x), list(self.versions[self.versions['model_id'] == self.model_id]['version'])))
        
        print(f'The {self.model_name} model has the following versions: {self.version_names}')
        return self.version_names
            
    def select_version(self, version):
        """
        Selects version.
        Parameters
        ----------
        version: string
            Version name.
        """
        self.version = version
        self.version_id = self.versions[self.versions['version'] == self.version].index[0]
        self.version_name = 'v'+ str(self.version)
        self.training_params =json.loads(self.versions[self.versions['version'] == self.version]['training_params'][self.version_id])
        self.image_ids = list(self.versions.iloc[self.version_id][['input_image_id', 'output_image_id']])
        self.collections = list(self.datasets.iloc[list(self.images.iloc[self.image_ids]['dataset_id'])]['slug'])
        self.bands = [self.training_params.get('in_bands'), self.training_params.get('out_bands')]
        self.scale, self.init_date, self.end_date = list(self.images.iloc[self.image_ids[0]][['scale', 'init_date', 'end_date']])
        
        print(f'Selected version name: {self.version_name}')
        print('Datasets: ', self.collections)
        print('Bands: ', self.bands)
        print('scale: ', self.scale)
        print('init_date: ', self.init_date)
        print('end_date: ', self.end_date)

class Predictor(object):
    """
    Prediction of Deep Learning models in Skydipper
    """
    def __init__(self):
        self.db_url = 'postgresql://postgres:postgres@0.0.0.0:5432/geomodels'
        self.engine = sqlalchemy.create_engine(self.db_url)
        self.table_names = self.engine.table_names()
        #self.datasets_api = Skydipper.Collection(search=' '.join(self.slugs_list), object_type=['dataset'], app=['skydipper'], limit=len(self.slugs_list))
        self.ee_tiles = 'https://earthengine.googleapis.com/map/{mapid}/{{z}}/{{x}}/{{y}}?token={token}' 
        self.datasets = df_from_query(self.engine, 'dataset') 
        self.images = df_from_query(self.engine, 'image')  
        self.models = df_from_query(self.engine, 'model') 
        self.versions = df_from_query(self.engine, 'model_versions') 
        self.bucket = 'geo-ai'
        self.project_id = 'skydipper-196010'
        ee.Initialize()

    def get_token(self, email):
        password = getpass.getpass('Skydipper login password:')

        payload = {
            "email": f"{email}",
            "password": f"{password}"
        }

        url = f'https://api.skydipper.com/auth/login'

        headers = {'Content-Type': 'application/json'}

        r = requests.post(url, data=json.dumps(payload), headers=headers)

        self.token = r.json().get('data').get('token')

    def select_model(self, model_name):
        """
        Selects model.
        Parameters
        ----------
        model_name: string
            Model name.
        """
        self.model_name = model_name
        self.model_id = self.models[self.models['model_name'] == model_name].index[0]
        self.model_type = self.models.iloc[self.model_id]['model_type']
        self.model_output = self.models.iloc[self.model_id]['model_output']
        self.version_names = list(map(lambda x: int(x), list(self.versions[self.versions['model_id'] == self.model_id]['version'])))
        
        print(f'The {self.model_name} model has the following versions: {self.version_names}')
        return self.version_names
            
    def select_version(self, version):
        """
        Selects version.
        Parameters
        ----------
        version: string
            Version name.
        """
        self.version = version
        self.version_id = self.versions[self.versions['version'] == self.version].index[0]
        self.version_name = 'v'+ str(self.version)
        self.training_params =json.loads(self.versions[self.versions['version'] == self.version]['training_params'][self.version_id])
        self.image_ids = list(self.versions.iloc[self.version_id][['input_image_id', 'output_image_id']])
        self.collections = list(self.datasets.iloc[list(self.images.iloc[self.image_ids]['dataset_id'])]['slug'])
        self.bands = [self.training_params.get('in_bands'), self.training_params.get('out_bands')]
        self.scale, self.init_date, self.end_date = list(self.images.iloc[self.image_ids[0]][['scale', 'init_date', 'end_date']])
        
        print(f'Selected version name: {self.version_name}')
        print('Datasets: ', self.collections)
        print('Bands: ', self.bands)
        print('scale: ', self.scale)
        print('init_date: ', self.init_date)
        print('end_date: ', self.end_date)

    def create_geostore_from_geojson(self, attributes, zoom=6):
        """Parse valid geojson into a geostore object and register it to a
        Gestore object on a server. 
        Parameters
        ----------
        attributes: list
            List of geojsons with the trainig, validation, and testing polygons.
        zoom: int
            A z-level for the map.
        """
        self.attributes = attributes
        # TODO(replace code with SkyPy)
        #self.polygon = Skydipper.Geometry(attributes=attributes) # This is here commented until SkyPy works again
        #self.geostore_id = self.polygon.id # This is here commented until SkyPy works again
        # Register geostore object on a server. Return the object, and instantiate a Geometry.
        if self.token:
            header= {
                'Authorization': 'Bearer ' + self.token,
                'Content-Type':'application/json'
                    }
            url = 'https://api.skydipper.com/v1/geostore'
            r = requests.post(url, headers=header, json=self.attributes)

            self.polygon = r.json().get('data').get('attributes')
            self.geostore_id = r.json().get('data').get('id')

        else:
            raise ValueError(f'Token is required use get_token() method first.')

        # Returns a folium map with the polygons
        self.features = self.polygon['geojson']['features']
        if len(self.features) > 0:
            shapely_geometry = [shape(feature['geometry']) for feature in self.features]
        else:
            shapely_geometry = None
    
        self.centroid = list(shapely_geometry[0].centroid.coords)[0][::-1]

        bbox = self.polygon.get('bbox')
        self.bounds = [bbox[2:][::-1], bbox[:2][::-1]] 
        ##bbox = self.multipolygon.attributes['bbox']
        ##self.bounds = [bbox[2:][::-1], bbox[:2][::-1]]        

        map = folium.Map(location=self.centroid, zoom_start=zoom)
        map.fit_bounds(self.bounds)

        self.nFeatures = len(self.features)
        self.colors = ['#64D1B8', 'red', 'blue']
        for n in range(self.nFeatures):
            style_function = lambda x: {
                'fillOpacity': 0.0,
                    'weight': 4,
                    'color': self.colors[0]
                    }
            folium.GeoJson(data=get_geojson_string(self.features[n]['geometry']), style_function=style_function,\
                 name='Polygon').add_to(map)
        
        map.add_child(folium.LayerControl())
        return map

    def predict_ai_platform(self, init_date=None, end_date=None, zoom=6):
        """
        Predict in AI Platform.
        Parameters
        ----------
        init_date: string
            Initial date of the composite.
        end_date: string
            Last date of the composite.
        """
        if init_date and end_date:
            self.init_date = init_date
            self.end_date = end_date
        else:
            self.init_date = self.init_date
            self.end_date = self.end_date

        self.kernel_size = int(self.versions['kernel_size'].iloc[self.version_id])
        self.input_image_id = self.versions.iloc[self.version_id]['input_image_id']
        values = json.loads(self.images.iloc[self.input_image_id]['bands_min_max'])
        # Create input composite
        self.image = ee_collection_specifics.Composite(self.collections[0])(self.init_date, self.end_date)
        # Normalize images
        if bool(values): 
            self.image = normalize_ee_images(self.image, self.collections[0], values)
        # Select bands and convert them into float
        self.image = self.image.select(self.bands[0]).float()

        # Output image
        if self.kernel_size == 1:
            input_tile_size = [1, 1]
            input_overlap_size = [0, 0]
        if self.kernel_size >1 :
            input_tile_size = [144, 144]
            input_overlap_size = [8, 8]

        # Load the trained model and use it for prediction.
        model = ee.Model.fromAiPlatformPredictor(
            projectName = self.project_id,
            modelName = self.model_name,
            version = self.version_name,
            inputTileSize = input_tile_size,
            inputOverlapSize = input_overlap_size,
            proj = ee.Projection('EPSG:4326').atScale(self.scale),
            fixInputProj = True,
            outputBands = {'prediction': {
                'type': ee.PixelType.float(),
                'dimensions': 1,
              }                  
            }
        )
        self.predictions = model.predictImage(self.image.toArray()).arrayFlatten([self.bands[1]])

        # Clip the prediction area with the polygon
        geometry = ee.Geometry.Polygon(self.polygon.get('geojson').get('features')[0].get('geometry').get('coordinates'))
        self.predictions  = self.predictions .clip(geometry)

        # Segmentate image:
        if self.model_output == 'segmentation':
            maxValues = self.predictions.reduce(ee.Reducer.max())

            self.predictions = self.predictions.addBands(maxValues)

            expression = ""
            for n, band in enumerate(bands[1]):
                expression = expression + f"(b('{band}') == b('max')) ? {str(n+1)} : "

            expression = expression + f"0"

            segmentation = self.predictions.expression(expression)
            self.predictions = self.predictions.addBands(segmentation.mask(segmentation).select(['constant'], ['categories']))
        
        # Use folium to visualize the input imagery and the predictions.
        mapid = self.image.getMapId({'bands': ee_collection_specifics.ee_bands_rgb(self.collections[0]), 'min': 0, 'max': 1})

        map = folium.Map(location=self.centroid, zoom_start=zoom)
        map.fit_bounds(self.bounds)

        folium.TileLayer(
            tiles=self.ee_tiles.format(**mapid),
            attr='Google Earth Engine',
            overlay=True,
            name='input median composite',
          ).add_to(map)

        for band in self.bands[1]:
            mapid = self.predictions.getMapId({'bands': [band], 'min': 0, 'max': 2})

            folium.TileLayer(
                tiles=self.ee_tiles.format(**mapid),
                attr='Google Earth Engine',
                overlay=True,
                name=band,
              ).add_to(map)

        if self.model_output == 'segmentation':
            mapid = self.predictions.getMapId({'bands': ['categories'], 'min': 1, 'max': len(bands[1])})

            folium.TileLayer(
                tiles=self.ee_tiles.format(**mapid),
                attr='Google Earth Engine',
                overlay=True,
                name='categories',
              ).add_to(map)

        for n in range(self.nFeatures):
            style_function = lambda x: {
                'fillOpacity': 0.0,
                    'weight': 4,
                    'color': self.colors[0]
                    }
            folium.GeoJson(data=get_geojson_string(self.features[n]['geometry']), style_function=style_function,\
                 name='Polygon').add_to(map)

        map.add_child(folium.LayerControl())

        return map





