import ee
import Skydipper
import sqlalchemy
import folium
import json
import time
import pandas as pd
import requests
import getpass
from shapely.geometry import shape
from google.cloud import storage
from google.cloud.storage import blob
from googleapiclient import discovery
from googleapiclient import errors
from oauth2client.client import GoogleCredentials

from .utils import df_from_query, df_to_db, df_to_csv, polygons_to_geoStoreMultiPoligon, get_geojson_string,\
    min_max_values, normalize_ee_images, get_image_ids, GeoJSONs_to_FeatureCollections, check_status_data,\
    removekey
import ee_collection_specifics

class Trainer(object):
    """
    Training and prediction of Deep Learning models in Skydipper
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
        self.datasets = self.get_table(table_name='dataset')
        self.images = self.get_table(table_name='image')
        self.models = self.get_table(table_name='model')
        self.versions = self.get_table(table_name='model_versions')
        self.bucket = 'geo-ai'
        self.project_id = 'skydipper-196010'
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

    def get_table(self, table_name='model_versions'):
        """
        Retrieve table from database.
        Parameters
        ----------
        table_name: string
            Table name
        """
        return df_from_query(self.engine, table_name)

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
            if not self.images[['dataset_id', 'scale', 'init_date', 'end_date', 'bands_min_max', 'norm_type']]\
                                .isin([dataset_id, self.scale, self.init_date, self.end_date, self.values[n], self.norm_type]).all(axis=1).any():
                # Append values to table
                dictionary = dict(zip(list(self.images.keys()), [[dataset_id], [''], [self.scale], [self.init_date], [self.end_date], [self.values[n]], [self.norm_type]]))
                self.images = self.images.append(pd.DataFrame(dictionary), ignore_index = True)

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
        df['training_params'] = df['training_params'].apply(lambda x : removekey(json.loads(x),'job_dir'))

        # Check if the version already exists
        if (df['training_params'] == self.training_params).any():
            # Get version id
            version_id = df[df['training_params'].apply(lambda x : removekey(x,'job_dir')) == self.training_params].index[0]

            # Check status
            status = df.iloc[version_id]['training_status']
            print('Version already exists with training status equal to:', status)

            if status == 'SUCCEEDED':
                print('The training job successfully completed.')
            if (status == 'CANCELLED') or (status == 'FAILED'):
                print(f'The training job was {status}.')
                if status == 'CANCELLED':  
                    print('Start training again.')
                if status == 'FAILED': 
                    print('Change training parameters and try again.')
                # Get training version
                self.training_version = df.iloc[version_id]['version']

                # Update job name
                job_name = 'job_v' + str(int(time.time()))

                # Add job directory
                self.training_params = json.loads(df.iloc[version_id]['training_params'])
                self.training_params['job_dir'] = 'gs://' + bucket + '/Models/' + str(self.model_id) + '/' +  str(self.training_version) + '/'

                # Save training version and clear status
                self.versions.at[version_id, 'training_params'] =  json.dumps(self.training_params)
                self.versions.at[version_id, 'training_status'] = ''

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
                version_id = df[df.all(axis=1)].index[0]

                self.versions.at[version_id, 'model_id'] = self.model_id
                self.versions.at[version_id, 'model_architecture'] = self.model_architecture
                self.versions.at[version_id, 'training_params'] = json.dumps(self.training_params)
                self.versions.at[version_id, 'version'] = self.training_version

            else:
                dictionary = dict(zip(list(self.versions.keys()), [[''], [''], [self.image_ids[0]], [self.image_ids[1]], [self.geostore_id], [self.kernel_size], [self.sample_size], [''], [''], ['COMPLETED'], [''], [''], ['']]))
                self.versions = self.versions.append(pd.DataFrame(dictionary), ignore_index = True, sort=False)
                version_id = self.versions.index[-1]

                self.versions.at[version_id, 'model_id'] = int(self.model_id)
                self.versions.at[version_id, 'model_architecture'] = self.model_architecture
                self.versions.at[version_id, 'training_params'] = json.dumps(self.training_params)
                self.versions.at[version_id, 'version'] = int(self.training_version)


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

        # Get a Python representation of the AI Platform Training services
        credentials = GoogleCredentials.from_stream(self.privatekey_path)
        ml = discovery.build('ml', 'v1', credentials = credentials)

        # Create a request to call projects.jobs.create.
        request = ml.projects().jobs().create(body=job_spec,
                      parent=project)

        # Make the call.
        try:
            response = request.execute()
            print(response)

        except errors.HttpError as err:
            # Something went wrong, print out some information.
            print('There was an error creating the training job. Check the details:')
            print(err._get_reason())
            
        # Save training status
        self.request_get = ml.projects().jobs().get(name=job_name,
              parent=project)
        self.response_get = self.request_get.execute()

        # TODO
        # Save image and model_versions tables
        df_to_csv(self.images, "image")
        df_to_csv(self.versions, "model_versions")
        df_to_db(self.images, self.engine, "image")
        df_to_db(self.versions, self.engine, "model_versions")






