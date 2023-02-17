import tensorflow as tf
import numpy as np
import os
import glob
import shutil
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#data loading
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz/"
zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)
base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

#creating labels
classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

#for cl in classes: