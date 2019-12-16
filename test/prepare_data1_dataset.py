from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import imageio
import rasterio
import glob
import h5py

    
def load_png_3bands(path):
    data_R = np.array([]) 
    data_G = np.array([]) 
    data_B = np.array([]) 
    for n, image_path in enumerate(glob.glob(path)):
        image = imageio.imread(image_path)
    
        data_R = np.append(data_R, image[:,:,0])
        data_G = np.append(data_G, image[:,:,1])
        data_B = np.append(data_B, image[:,:,2])

    data_R = data_R.reshape((n+1, image.shape[0], image.shape[1]))
    data_G = data_G.reshape((n+1, image.shape[0], image.shape[1]))
    data_B = data_B.reshape((n+1, image.shape[0], image.shape[1]))
    
    data = np.stack((data_R,data_G,data_B), axis=2)
    data = np.moveaxis(data, 2, 3)
    
    # Normalize
    data = data.astype('float32')
    data /= 255
        
    return data
    
def load_png_1band(path):
    data = np.array([])
    for n, image_path in enumerate(glob.glob(path)):
        image = imageio.imread(image_path)
    
        data = np.append(data, image)
    data = data.reshape((n+1, image.shape[0], image.shape[1]))
    
    return data

if (__name__ == '__main__'):
    # Load data
    # Validation data
    x_validation = load_png_3bands("/Users/ikersanchez/Vizzuality/PROIEKTUAK/Skydipper/Data/dataset1/images_prepped_test/*.png")
    y_validation = load_png_1band("/Users/ikersanchez/Vizzuality/PROIEKTUAK/Skydipper/Data/dataset1/annotations_prepped_test/*.png")
    
    # Train data
    x_train = load_png_3bands("/Users/ikersanchez/Vizzuality/PROIEKTUAK/Skydipper/Data/dataset1/images_prepped_train/*.png")
    y_train = load_png_1band("/Users/ikersanchez/Vizzuality/PROIEKTUAK/Skydipper/Data/dataset1/annotations_prepped_train/*.png")
    
    # Visualize data
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    ax = axs[0]
    ax.imshow(x_train[0,:,:,:])
    
    ax = axs[1]
    ax.imshow(y_train[0,:,:])
    
    plt.show()
    
    # Preprocess class labels for Keras
    # Convert 1-dimensional class arrays to 12-dimensional class matrices
    y_train = np_utils.to_categorical(y_train, num_classes=12)
    y_validation = np_utils.to_categorical(y_validation, num_classes=12)
    
    # Save samples
    f = h5py.File("/Users/ikersanchez/Vizzuality/PROIEKTUAK/Skydipper/Work/cnn-models/SegNet/Samples/data1/Samples/data1/x_train.h5", 'w')
    f.create_dataset('x_train', data=x_train)     
    f.close()
    
    f = h5py.File("/Users/ikersanchez/Vizzuality/PROIEKTUAK/Skydipper/Work/cnn-models/SegNet/Samples/data1/Samples/data1/y_train.h5", 'w')
    f.create_dataset('y_train', data=y_train)     
    f.close()
    
    f = h5py.File("/Users/ikersanchez/Vizzuality/PROIEKTUAK/Skydipper/Work/cnn-models/SegNet/Samples/data1/Samples/data1/x_validation.h5", 'w')
    f.create_dataset('x_validation', data=x_validation)     
    f.close()
    
    f = h5py.File("/Users/ikersanchez/Vizzuality/PROIEKTUAK/Skydipper/Work/cnn-models/SegNet/Samples/data1/Samples/data1/y_validation.h5", 'w')
    f.create_dataset('y_validation', data=y_validation)     
    f.close()