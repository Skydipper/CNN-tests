import ee
import json
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from google.oauth2 import service_account
from googleapiclient import discovery
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib as mpl

account = 'skydipper@skydipper-196010.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(account, 'privatekey_EE.json')
ee.Initialize(credentials)

def predict_json(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the AI Platform Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """    
    # To authenticate set the GOOGLE_APPLICATION_CREDENTIALS
    credentials = service_account.Credentials.from_service_account_file('privatekey_ML_Engine.json')
    
    # Create the AI Platform service object.
    service = discovery.build('ml', 'v1', credentials=credentials)
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']

def max_pixels_class(x):
    """Binarize the output"""
    x_new = x*0
    max_val = np.amax(x, axis=2)
    size = x.shape
    for i in range(size[-1]):
        ima = x[:,:,i]*0
        ima[np.where(x[:,:,i] == max_val)] = 1
        x_new[:,:,i]= ima
        
    x_new = np.argmax(x_new, axis=2)

    return x_new

def segmented_image(arr):
    """Get segmented image as a numpy array"""
    
    colors_dic = {'0':'#ffd300', '1':'#93cc93', '2':'#4970a3', '3':'#999999'}

    keys = list(np.unique(arr))
    keys = [str(i) for i in keys]
    colors = [colors_dic.get(key) for key in keys]
    
    cmap = mpl.colors.ListedColormap(colors)

    fig = Figure()
    fig.set_size_inches(256/fig.get_dpi(), 256/fig.get_dpi())
    fig.subplots_adjust(0,0,1,1)
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)

    ax.imshow(arr, cmap=cmap)
    ax.axis('off')
    ax.margins(0,0)

    canvas.draw()       # draw the canvas, cache the renderer

    width, height = fig.get_size_inches() * fig.get_dpi()

    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    
    return image
    
def segmentate(request):
    request = request.get_json()
    
    source = request.get('source')
    band_viz = request.get('band_viz')
    
    collection = source.split("/")[0]

    img = ee.Image(source).visualize(**band_viz)
    info = img.getInfo()
    coordinates = np.array(info.get('properties').get('system:footprint').get('coordinates'))

    #Get tile
    if collection == 'COPERNICUS':
        img = ee.Image(source).divide(10000).visualize(**band_viz)
        tile_rgb = img.getThumbUrl({'dimensions':[256,256]})
    else:
        tile_rgb = img.getThumbUrl({'dimensions':[256,256]})
        
    response = requests.get(tile_rgb)
    img_rgb = Image.open(BytesIO(response.content))
    
    # Convert it into numpy array
    arr_rgb = np.array(img_rgb)[:,:,:3]
    
    # Create instance
    data = np.around(arr_rgb/255,2).tolist()
    instance = {"image" : data}
    
    # Request prediction
    response = predict_json(project="skydipper-196010", model="segnet", instances=instance, version="v1")
    
    # Convert output into numpy array
    output = np.array(response[0].get('output'))
    
    # Binarize the output and convert it into classes
    output = max_pixels_class(output)
    
    # Get segmented image as a numpy array
    image = segmented_image(output)
    
    # Get bounding box
    bbox = [min(coordinates[:,0]), min(coordinates[:,1]), max(coordinates[:,0]), max(coordinates[:,1])]

        
    return json.dumps({'output': image.tolist(), 'bbox': bbox})