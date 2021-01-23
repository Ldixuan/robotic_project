import os
import numpy as np
import pandas as pd
# import seaborn as sns
import cv2
from pylab import rcParams
# from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
import albumentations as A
import random

BOX_COLOR = (255, 0, 0)


def visualize_bbox(img, bbox, color=BOX_COLOR, thickness=2):
    xmin, ymin, xmax, ymax = map(lambda v: int(v), bbox)

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=color, thickness=thickness)
    return img


def show_image(image, bbox):
    image = visualize_bbox(image.copy(), bbox)
    f = plt.figure(figsize=(18, 12))
    plt.imshow(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        interpolation='nearest'
    )
    plt.axis('off')
    f.tight_layout()
    plt.show()


def show_augmented(augmentation, image, bbox):
    augmented = augmentation(image=image, bboxes=[bbox], field_id=['1'])
    show_image(augmented['image'], augmented['bboxes'][0])


bbox_params = A.BboxParams(
    format='pascal_voc',
    min_area=1,
    min_visibility=0.5,
    label_fields=['field_id']
)

doc_aug = A.Compose([
    A.Flip(p=0.25),
    A.RandomGamma(gamma_limit=(20, 300), p=0.5),
    A.RandomBrightnessContrast(p=0.85),
    A.Rotate(limit=35, p=0.9),
    A.RandomRotate90(p=0.25),
    A.RGBShift(p=0.75),
    A.GaussNoise(p=0.25)
], bbox_params=bbox_params)

DEST_PATH = 'data/aug'
IMAGES_PATH = 'data/origin/Stop sign'
LABLE_PATH = 'data/origin/Stop sign/Label'

os.makedirs(DEST_PATH, exist_ok=True)
os.makedirs(IMAGES_PATH, exist_ok=True)

images_names = []
for f in os.listdir(IMAGES_PATH):
    ext = os.path.splitext(f)[1]
    if ext.lower() != '.jpg':
        continue
    images_names.append(f)

rows = []
for img_name in images_names:
    print(img_name)

    img = cv2.imread(f"{IMAGES_PATH}/{img_name}")
    label_f = open(f"{LABLE_PATH}/{img_name[:-4]}.txt", 'r')
    label_text = label_f.readlines()
    label_f.close()

    bbox_i = [t[:-1].split(" ")[2:] for t in label_text]

    stop_sign_box = [[float(i) for i in l] for l in bbox_i]

    for bbox in stop_sign_box:
        xmin, ymin, xmax, ymax = map(lambda v: int(v), bbox)

        height, width, channels = img.shape

        rows.append({
                'filename': img_name,
                'width': width,
                'height': height,
                'class': 'Stop sign',
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })

    cv2.imwrite(f'{DEST_PATH}/{img_name}', img)

    # show_image(img, stop_sign_box[0])

    for i in range(10):
        augmented = doc_aug(image=img, bboxes=stop_sign_box, field_id=[1 for i in stop_sign_box])

        if not augmented['bboxes']:
            continue

        file_name = f'{img_name[:-4]}_aug_{i}.jpg'
        height, width, channels = augmented['image'].shape

        for bbox in augmented['bboxes']:
            xmin, ymin, xmax, ymax = map(lambda v: int(v), bbox)

            rows.append({
                'filename': file_name,
                'width': width,
                'height': height,
                'class': 'Stop sign',
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })

        cv2.imwrite(f'{DEST_PATH}/{file_name}', augmented['image'])

pd.DataFrame(rows).to_csv('data/annotation/annotations.csv', header=True, index=None)
