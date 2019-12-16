import ee
import numpy as np
import pandas as pd
import math
import h5py


def normalize_data(data):
    
    dim = data.shape
    
    band_max = []
    band_min = []
    for n in range(dim[-1]):
        mx = data[:,:,:,n].max()
        mn = data[:,:,:,n].min()
        
        band_max.append(mx)
        band_min.append(mn)
        
        data[:,:,:,n] = (data[:,:,:,n]-mn)/(mx-mn)
    
    return data, band_max, band_min 

def replace_values(array, class_labels, new_label):
    array_new = np.copy(array)
    for i in range(len(class_labels)):
        array_new[np.where(array == class_labels[i])] = new_label
        
    return array_new

def categorical_data(data):
    # Area of Interest (AoI)
    point = [-120.7224, 37.3872]
    geom = ee.Geometry.Point(point).buffer(100)
    # Start and stop of time series
    startDate = ee.Date('2016')
    stopDate  = ee.Date('2017')
    # Read the ImageCollection
    dataset = ee.ImageCollection('USDA/NASS/CDL')\
        .filterBounds(geom)\
        .filterDate(startDate,stopDate)
    # Get the cropland class values and names
    cropland_info = pd.DataFrame({'cropland_class_values':dataset.getInfo().get('features')[0].get('properties').get('cropland_class_values'),
                              'cropland_class_palette':dataset.getInfo().get('features')[0].get('properties').get('cropland_class_palette'),
                              'cropland_class_names':dataset.getInfo().get('features')[0].get('properties').get('cropland_class_names')
                             })
    

    # New classes
    land = ['Shrubland', 'Barren', 'Grassland/Pasture', 'Deciduous Forest', 'Evergreen Forest', 'Mixed Forest', 'Wetlands', 'Woody Wetlands', 'Herbaceous Wetlands']
    water = ['Water', 'Open Water', 'Aquaculture']
    urban = ['Developed', 'Developed/Open Space', 'Developed/High Intensity', 'Developed/Low Intensity', 'Developed/Med Intensity']

    class_labels_0 = np.array(cropland_info[cropland_info['cropland_class_names'].isin(land)]['cropland_class_values'])
    class_labels_1 = np.array(cropland_info[cropland_info['cropland_class_names'].isin(water)]['cropland_class_values'])
    class_labels_2 = np.array(cropland_info[cropland_info['cropland_class_names'].isin(urban)]['cropland_class_values'])
    class_labels_3 = np.array(cropland_info[(~cropland_info['cropland_class_names'].isin(land)) & 
                                        (~cropland_info['cropland_class_names'].isin(water)) & 
                                        (~cropland_info['cropland_class_names'].isin(urban))]['cropland_class_values'])

    # We replace the class labels
    new_data = np.copy(data[:,:,:,0])
    new_data = replace_values(new_data, class_labels_3, 3.)
    new_data = replace_values(new_data, class_labels_2, 2.)
    new_data = replace_values(new_data, class_labels_1, 1.)
    new_data = replace_values(new_data, class_labels_0, 0.)

    # Convert 1-dimensional class arrays to 4-dimensional class matrices
    from keras.utils import np_utils
    new_data = np_utils.to_categorical(new_data, 4)
    
    return new_data


def train_validation_split(x, y, val_size=20):
    t=x.shape[0]
    size = int(t*((100-val_size)/100))
    
    xt = x[:size,:]
    xv = x[size:,:]
    yt = y[:size,:]
    yv = y[size:,:]
    
    return xt, xv, yt, yv

def write_data(output_path, name, cube):
    #Write output parameters
    h5f = h5py.File(output_path, 'w')
    h5f.create_dataset(name, data=cube)  
    h5f.close()
    
    
def max_pixels(x):
    """
    Binarize the output taking the highest pixel value
    """
    x_new = x*0
    max_val = np.amax(x, axis=2)
    size = x.shape
    for i in range(size[-1]):
        ima = x[:,:,i]*0
        ima[np.where(x[:,:,i] == max_val)] = 1
        x_new[:,:,i]= ima

    return x_new