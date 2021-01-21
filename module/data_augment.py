from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os

dataset_dir = 'data/origin'
image_filenames = [os.path.join(dataset_dir, x) for x in os.listdir(dataset_dir) if x.endswith(".jpg")]

datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.15,
                             zoom_range=0.1, channel_shift_range=10, horizontal_flip=True)

datagen = datagen.flow_from_directory(dataset_dir, target_size=(200, 200), save_to_dir="data/aug",
                                      class_mode='binary', save_prefix='aug', save_format='jpg', batch_size=10)

for x, val in zip(datagen, range(500)):
    print(val)

