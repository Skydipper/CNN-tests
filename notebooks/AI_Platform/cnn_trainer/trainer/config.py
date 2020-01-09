
import tensorflow as tf

from . import env

# Define your Google Cloud Storage bucket
bucket = env.bucket_name

# Specify names of output locations in Cloud Storage.
dataset_name = 'Landsat8_Impervious'#'Landsat8_Cropland'
job_dir = 'gs://' + bucket + '/' + 'cnn-models/'+ dataset_name +'/trainer'
model_dir = job_dir + '/model'
logs_dir = job_dir + '/logs'

# Pre-computed training and eval data.
base_names = ['training_patches', 'eval_patches']
folder = 'cnn-models/'+dataset_name+'/data'

# Specify inputs/outputs to the model
in_bands = ['B1','B2','B3','B4','B5','B6','B7']
out_bands = ['impervious']#['cropland', 'land', 'water', 'urban']

# Specify the size and shape of patches expected by the model.
kernel_size = 256

# Sizes of the training and evaluation datasets.
train_size = 1000*47
eval_size = 1000*12

# Specify model training parameters.
model_type = 'regression'
model_architecture = 'deepvel'
batch_size = 16
epochs = 5
shuffle_size = 2000
learning_rate = 1e-3
optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
loss = 'mse'
metrics = ['accuracy']