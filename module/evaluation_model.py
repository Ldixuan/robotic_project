import tensorflow as tf
import os
import re
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

path_to_pb = "exported-models/my_model/saved_model"
path_to_test_image_dir = "data/test_image"
path_to_labels = "data/annotation/label_map.pbtxt"



def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))




if __name__ == '__main__':
    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)
    detect_fn = tf.saved_model.load(path_to_pb)

    images = [f for f in os.listdir(path_to_test_image_dir)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]

    for img_path in images:
        image_np = load_image_into_numpy_array(f"data/test_image/{img_path}")
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
              image_np_with_detections,
              detections['detection_boxes'],
              detections['detection_classes'],
              detections['detection_scores'],
              category_index,
              use_normalized_coordinates=True,
              max_boxes_to_draw=200,
              min_score_thresh=.30,
              agnostic_mode=False)

        plt.figure()
        plt.imshow(image_np_with_detections)
        plt.savefig(f"data/test_image_evaluation/{img_path}.png")
        print('Done')


