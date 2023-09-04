from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential

dataset_dir = Path('dataset')
image_count = len(list(dataset_dir.glob('*/*.jpg')))
print(f'Total images {image_count}')

batch_size = 32
img_width = 180
img_height = 180

train_ds = keras.utils.image_dataset_from_directory(dataset_dir,
                                                    validation_split=0.2,
                                                    subset='training',
                                                    seed=123,
                                                    image_size=(img_height, img_width),
                                                    batch_size=batch_size)

val_ds = keras.utils.image_dataset_from_directory(dataset_dir,
                                                  validation_split=0.2,
                                                  subset='validation',
                                                  seed=123,
                                                  image_size=(img_height, img_width),
                                                  batch_size=batch_size)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds= val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create model

num_classes = len(class_names)
model =Sequential(
    layers.experimental.preprocessing.Rescaling(1/255,input_shape=(img_height,img_width,3)),

    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D()

)
