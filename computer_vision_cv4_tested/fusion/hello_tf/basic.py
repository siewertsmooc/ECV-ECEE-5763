# Based on - https://www.tensorflow.org/tfx/tutorials/serving/rest_simple
#
# Please see the above reference for more information
#
# You may need to install Python support packages as follows
#
#pip3 install grpcio
#pip3 install matplotlib
#pip3 install tensorrt
#pip3 install nvidia-pyindex
#pip3 install nvidia-cudnn
#
# Adapted by Sam Siewert, 1/16/2025
#

import sys

############################### Make sure Python environment is sane

# Confirm that we're using Python 3
assert sys.version_info.major == 3, 'Oops, not running Python 3. Use Runtime > Change runtime type'

import tensorflow as tf
from tensorflow import keras

print("Avoid memory hogging")
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

# TensorFlow and tf.keras
print("Installing dependencies for Colab environment")

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

print('\nTensorFlow version: {}'.format(tf.__version__))


################################# Get the data to train the model

print('\nGetting Fashion MNIST dataset')

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale the values to 0.0 to 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape for feeding into the model
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))


################################# Train and evaluate a Keras model

print('\nCreating Keras model')

model = keras.Sequential([
  keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3, 
                      strides=2, activation='relu', name='Conv1'),
  keras.layers.Flatten(),
  keras.layers.Dense(10, name='Dense')
])
model.summary()

######## Exit here if you just want to see the created model summary
#exit()

testing = False
epochs = 5

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

print('\n model compiled!')

######## Exit here if you just want to compile the model and not train it
#exit()


model.fit(train_images, train_labels, epochs=epochs)

print('\n model trained!')

######## Exit here if you just want to train the model and not test
#exit()

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy: {}'.format(test_acc))

print('\n model tested!')



################################# Save Keras model
#
# Simplified to just use a path and name in /tmp
#
import tempfile

#MODEL_DIR = tempfile.gettempdir()
MODEL_DIR = './'
export_path = os.path.join(MODEL_DIR, 'basic.keras')
print('export_path = {}\n'.format(export_path))

tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True
)

print('\nSaved model as ./basic.keras')
#print('\nSaved model as /tmp/basic.keras')

################################# Now you can serve this saved model as documented in reference
#
# https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/saved_model.md#cli-to-inspect-and-execute-savedmodel
#
# https://keras.io/2.16/api/models/model_saving_apis/model_saving_and_loading/
#
#loaded_model = tf.keras.models.load_model("/tmp/basic.keras")
loaded_model = tf.keras.models.load_model("./basic.keras")

print('\nSaved model has been reloaded')
