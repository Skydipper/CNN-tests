import ee; ee.Initialize()
import numpy as np
import pandas as pd
import math
import h5py
import os
from preprocess import ee_collection_specifics

class preprocess_datasets:
    
    def __init__(self, dataset_names, collections):
        """
        Class used to get the datasets from Earth Engine
        Parameters
        ----------
        dataset_names: array of strings
            Input and output h5py dataset names. Example: ['data_x', 'data_y']
        collections: array of strings
            Input and output gee collection names. Example: ['Sentinel2', 'CroplandDataLayers']

        """
        
        self.dataset_names = dataset_names
        self.collections = collections
        
        # h5py file dtypes
        self.h5py_dtype_x = ee_collection_specifics.h5py_dtype(self.collections[0])
        self.h5py_dtype_y = ee_collection_specifics.h5py_dtype(self.collections[1])
        
        # Function to change the class labels 
        self.class_labels = ee_collection_specifics.change_class_labels(self.collections[1])
 
        # Number of classes
        self.nClasses = ee_collection_specifics.nClasses(self.collections[1])
        
        # Path of the files.
        self.path = './samples/'

    def normalization_values(self):
        
        ## Read input dataset
        with h5py.File(self.path+self.dataset_names[0]+'.hdf5', 'r') as f:
            data = f[self.dataset_names[0]]
    
            dim = data.shape
    
            min_v = []
            max_v = []
                
            for n in range(dim[-1]):
                min_v.append(data[:,:,:,n].min())
                max_v.append(data[:,:,:,n].max())
    
            ## Save max min values 
            np.savez(self.path+'normalization_values', min_v, max_v)
            
    def change_class_labels(self):
        if self.nClasses:
            f = h5py.File(self.path+self.dataset_names[1]+'.hdf5', 'r')
            data_y = f[self.dataset_names[1]]
        
            dim = list(data_y.shape)
            dim[-1] = self.nClasses
    
            with h5py.File(self.path+self.dataset_names[1]+'_classes'+'.hdf5', 'w') as f:
                data = f.create_dataset(self.dataset_names[1], dim, dtype=self.h5py_dtype_y)
                
                n = 10000
                for i in range(math.ceil(dim[0]/n)):
                    data[i*n:(i+1)*n,:,:,:] = ee_collection_specifics.change_class_labels(self.collections[1])(data_y[i*n:(i+1)*n,:,:,:])
            
            f.close()
        
        ## Remove input file
        os.remove(self.path+self.dataset_names[1]+'.hdf5')
        
        ## Rename output file
        os.rename(self.path+self.dataset_names[1]+'_classes'+'.hdf5',self.path+self.dataset_names[1]+'.hdf5')
    
    def randomize_datasets(self):
        
        fx = h5py.File(self.path+self.dataset_names[0]+'.hdf5', 'a')
        data_x = fx[self.dataset_names[0]]

        fy = h5py.File(self.path+self.dataset_names[1]+'.hdf5', 'a')
        data_y = fy[self.dataset_names[1]]

        arr_t = np.arange(data_x.shape[0])
        np.random.shuffle(arr_t)

        for t in range(len(arr_t)):
            data_x[t,:] = data_x[arr_t[t],:]
            data_y[t,:] = data_y[arr_t[t],:]

        fx.close()
        fy.close()


    def train_validation_split(self, val_size=20):
        fx = h5py.File(self.path+self.dataset_names[0]+'.hdf5', 'r')
        data_x = fx[self.dataset_names[0]]
        
        fy = h5py.File(self.path+self.dataset_names[1]+'.hdf5', 'r')
        data_y = fy[self.dataset_names[1]]
        
        t = data_x.shape[0]
        size = int(t*((100-val_size)/100))
        
        dimx_train = list(data_x.shape)
        dimx_val = list(data_x.shape)    
        dimx_train[0] = size
        dimx_val[0] = t-size
        
        dimy_train = list(data_y.shape)
        dimy_val = list(data_y.shape)    
        dimy_train[0] = size
        dimy_val[0] = t-size
    
        with h5py.File(self.path+'x_train'+'.hdf5', 'w') as f:
            data = f.create_dataset('x_train', dimx_train, chunks=True, dtype=self.h5py_dtype_x)

            data[:] = data_x[:size,:]
        
        with h5py.File(self.path+'x_validation'+'.hdf5', 'w') as f:
            data = f.create_dataset('x_validation', dimx_val, chunks=True, dtype=self.h5py_dtype_x)

            data[:] = data_x[size:,:]
        
        with h5py.File(self.path+'y_train'+'.hdf5', 'w') as f:
            data = f.create_dataset('y_train', dimy_train, chunks=True, dtype=self.h5py_dtype_y)

            data[:] = data_y[:size,:]
        
        with h5py.File(self.path+'y_validation'+'.hdf5', 'w') as f:
            data = f.create_dataset('y_validation', dimy_val, chunks=True, dtype=self.h5py_dtype_y)

            data[:] = data_y[size:,:]
    
        fx.close()
        fy.close()
    

                
                
            