
from google.cloud import storage
from google.cloud.storage import blob
import json

from .models.CNN.regression import deepvel as CNNregDeepVel, segnet as  CNNregSegNet, unet as CNNregUNet
from .models.CNN.segmentation import deepvel as CNNsegDeepVel, segnet as  CNNsegSegNet, unet as CNNsegUNet
from .models.MLP.regression import sequential1 as MLPregSequential1

def select_model(path):
    # Read training parameters from GCS
    client = storage.Client(project='skydipper-196010')
    bucket = client.get_bucket('geo-ai')
    blob = bucket.blob(path)
    config = json.loads(blob.download_as_string(client=client).decode('utf-8'))
    
    # Model's dictionary
    models = {'CNN':
              {
                  'regression': 
                  {
                      'deepvel': CNNregDeepVel.create_keras_model,
                      'segnet': CNNregSegNet.create_keras_model,
                      'unet': CNNregUNet.create_keras_model,
                  },
                  'segmentation': 
                  {
                      'deepvel': CNNsegDeepVel.create_keras_model,
                      'segnet': CNNsegSegNet.create_keras_model,
                      'unet': CNNsegUNet.create_keras_model,
                  }
              }, 
              'MLP': 
              {
                  'regression': 
                  {
                      'sequential1': MLPregSequential1.create_keras_model,
                  }
              }
             }
    
    return models.get(config.get('model_type')).get(config.get('model_output')).get(config.get('model_architecture'))
