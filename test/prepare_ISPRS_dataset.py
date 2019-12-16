from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import imageio
import rasterio
import glob
import h5py
import os
import argparse

    
def load_tif_3bands(path, files):
    data_R = np.array([]) 
    data_G = np.array([]) 
    data_B = np.array([]) 
    for n, file in enumerate(files):
        image_path = path+file
        image = rasterio.open(image_path)
    
        data_R = np.append(data_R, image.read(1))
        data_G = np.append(data_G, image.read(2))
        data_B = np.append(data_B, image.read(3))

    data_R = data_R.reshape((n+1, image.read(1).shape[0], image.read(1).shape[1]))
    data_G = data_G.reshape((n+1, image.read(1).shape[0], image.read(1).shape[1]))
    data_B = data_B.reshape((n+1, image.read(1).shape[0], image.read(1).shape[1]))
    
    data = np.stack((data_R,data_G,data_B), axis=2)
    data = np.moveaxis(data, 2, 3)
    
    # Normalize
    data = data.astype('float32')
    data /= 255
        
    return data

def from_rgb_to_categorical(array):
    """
    The six categories/classes have been defined as:
        1. Impervious surfaces (RGB: 255, 255, 255)
        2. Building (RGB: 0, 0, 255)
        3. Low vegetation (RGB: 0, 255, 255)
        4. Tree (RGB: 0, 255, 0)
        5. Car (RGB: 255, 255, 0)
        6. Clutter/background (RGB: 255, 0, 0)
    """
    t = array.shape[0]
    y = array.shape[1]
    x = array.shape[2]
    c = array.shape[3]
    
    image = np.zeros((t,y,x))
    
    # 0. Impervious surfaces (RGB: 255, 255, 255)
    image[np.where((array[:,:,:,0] == 1.) & (array[:,:,:,1] == 1.) & (array[:,:,:,2] == 1.))] = 0.
    # 1. Building (RGB: 0, 0, 255)
    image[np.where((array[:,:,:,0] == 0.) & (array[:,:,:,1] == 0.) & (array[:,:,:,2] == 1.))] = 1.
    # 2. Low vegetation (RGB: 0, 255, 255)
    image[np.where((array[:,:,:,0] == 0.) & (array[:,:,:,1] == 1.) & (array[:,:,:,2] == 1.))] = 2.
    # 3. Tree (RGB: 0, 255, 0)
    image[np.where((array[:,:,:,0] == 0.) & (array[:,:,:,1] == 1.) & (array[:,:,:,2] == 0.))] = 3.
    # 4. Car (RGB: 255, 255, 0)
    image[np.where((array[:,:,:,0] == 1.) & (array[:,:,:,1] == 1.) & (array[:,:,:,2] == 0.))] = 4.
    # 5. Clutter/background (RGB: 255, 0, 0)
    image[np.where((array[:,:,:,0] == 1.) & (array[:,:,:,1] == 0.) & (array[:,:,:,2] == 0.))] = 5.
    
    # Convert 1-dimensional class arrays to 6-dimensional class matrices
    image = np_utils.to_categorical(image, num_classes=6)
    return image

def subfield(cube, xr, yr):
    #Subfield selection
    cube_sub = cube[:,yr[0]:yr[1],xr[0]:xr[1],:]
    return cube_sub
    
def write_data(output_path, name, cube):
    #Write output parameters
    h5f = h5py.File(output_path, 'w')
    h5f.create_dataset(name, data=cube)  
    h5f.close()

def resize_patches(x_train, y_train, x_validation, y_validation, patch_size = 200):
    stt, sty, stx, stz = x_train.shape
    svt, svy, svx, svz = y_validation.shape
    
    num_pathces_per_frame = sty/patch_size*stx/patch_size
    xt = np.zeros((int(stt*num_pathces_per_frame),patch_size,patch_size,int(stz)), dtype=np.float32)
    yt = np.zeros((int(stt*num_pathces_per_frame),patch_size,patch_size,int(svz)), dtype=np.float32)
    xv = np.zeros((int(svt*num_pathces_per_frame),patch_size,patch_size,int(stz)), dtype=np.float32)
    yv = np.zeros((int(svt*num_pathces_per_frame),patch_size,patch_size,int(svz)), dtype=np.float32)

    n=0
    for i in np.arange(sty/patch_size):
        for j in np.arange(stx/patch_size):
            xr=[int(patch_size*i),int(patch_size+patch_size*i)]
            yr=[int(patch_size*j),int(patch_size+patch_size*j)]

            xt[(stt*n):(stt+stt*n),:,:,:] = subfield(x_train,xr,yr)
            yt[(stt*n):(stt+stt*n),:,:,:] = subfield(y_train,xr,yr)
            xv[(svt*n):(svt+svt*n),:,:,:] = subfield(x_validation,xr,yr)
            yv[(svt*n):(svt+svt*n),:,:,:] = subfield(y_validation,xr,yr)
            n=n+1
    return xt, yt, xv, yv

if (__name__ == '__main__'):
    # Load data
    directory_x = "/Users/ikersanchez/Vizzuality/PROIEKTUAK/Skydipper/Data/Potsdam/2_Ortho_RGB/"
    directory_y = "/Users/ikersanchez/Vizzuality/PROIEKTUAK/Skydipper/Data/Potsdam/5_Labels_all/"
    files_x_train = sorted(os.listdir(directory_x))[:8]
    files_y_train = sorted(os.listdir(directory_y))[:8]
    files_x_validation = sorted(os.listdir(directory_x))[8:10]
    files_y_validation = sorted(os.listdir(directory_y))[8:10]
    # Train data
    x_train = load_tif_3bands(directory_x, files_x_train)
    y_train = load_tif_3bands(directory_y, files_y_train)
    # Validation data
    x_validation = load_tif_3bands(directory_x, files_x_validation)
    y_validation = load_tif_3bands(directory_y, files_y_validation)

    # Visualize data
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    ax = axs[0]
    ax.imshow(x_train[0,:,:,:])

    ax = axs[1]
    ax.imshow(y_train[0,:,:,:])

    plt.show()

    # Preprocess class labels for Keras
    y_train = from_rgb_to_categorical(y_train)
    y_validation = from_rgb_to_categorical(y_validation)


    # We extract patches of 200×200 pixels
    patch_size = 200
    xt, yt, xv, yv = resize_patches(x_train, y_train, x_validation, y_validation, patch_size = 200)

    # We randomize the datasets
    stt=xt.shape[0]
    svt=xv.shape[0]
    arr_t = np.arange(stt)
    arr_v = np.arange(svt)
    np.random.shuffle(arr_t)
    np.random.shuffle(arr_v)

    x_train = xt[arr_t,:,:,:]
    y_train = yt[arr_t,:,:,:]
    x_validation = xv[arr_v,:,:,:]
    y_validation = yv[arr_v,:,:,:]

    # Save samples
    write_data("/Users/ikersanchez/Vizzuality/PROIEKTUAK/Skydipper/Work/cnn-models/SegNet/Samples/Potsdam/x_train.h5", 'x_train', x_train)

    write_data("/Users/ikersanchez/Vizzuality/PROIEKTUAK/Skydipper/Work/cnn-models/SegNet/Samples/Potsdam/y_train.h5", 'y_train', y_train)

    write_data("/Users/ikersanchez/Vizzuality/PROIEKTUAK/Skydipper/Work/cnn-models/SegNet/Samples/Potsdam/x_validation.h5", 'x_validation', x_validation)

    write_data("/Users/ikersanchez/Vizzuality/PROIEKTUAK/Skydipper/Work/cnn-models/SegNet/Samples/Potsdam/y_validation.h5", 'y_validation', y_validation)
