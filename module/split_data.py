import os
import re
from shutil import copyfile
import math
import random
import pandas as pd

SOURCE_PATH = "data/aug"
DEST_PATH = "data"
TRAIN_DIR = os.path.join(DEST_PATH, 'train')
TEST_DIR = os.path.join(DEST_PATH, 'test')

ratio = 0.1

if not os.path.exists(TRAIN_DIR):
    os.makedirs(TRAIN_DIR)
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)

annotations = pd.read_csv('data/annotation/annotations.csv')

rows_test = []
rows_train = []

images = [f for f in os.listdir(SOURCE_PATH)
          if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]

num_images = len(images)
num_test_images = math.ceil(ratio*num_images)

for i in range(num_test_images):
    idx = random.randint(0, len(images)-1)
    filename = images[idx]
    copyfile(os.path.join(SOURCE_PATH, filename),
             os.path.join(TEST_DIR, filename))

    print(filename)
    annotation = annotations.loc[annotations['filename'] == filename].values[0].tolist()

    rows_test.append({
                'filename': annotation[0],
                'width': annotation[1],
                'height': annotation[2],
                'class': annotation[3],
                'xmin': annotation[4],
                'ymin': annotation[5],
                'xmax': annotation[6],
                'ymax': annotation[7]
            })

    images.remove(images[idx])

for filename in images:
    copyfile(os.path.join(SOURCE_PATH, filename),
             os.path.join(TRAIN_DIR, filename))

    annotation = annotations.loc[annotations['filename'] == filename].values[0].tolist()

    rows_train.append({
                'filename': annotation[0],
                'width': annotation[1],
                'height': annotation[2],
                'class': annotation[3],
                'xmin': annotation[4],
                'ymin': annotation[5],
                'xmax': annotation[6],
                'ymax': annotation[7]
            })

pd.DataFrame(rows_test).to_csv('data/annotation/annotations_test.csv', header=True, index=None)
pd.DataFrame(rows_train).to_csv('data/annotation/annotations_train.csv', header=True, index=None)

