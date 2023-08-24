import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import time
from contour import contour

done = time.time()

""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)

""" Global parameters """
global IMG_H
global IMG_W
global NUM_CLASSES
global CLASSES
global COLORMAP

""" Hyperparameters """
IMG_H = 320
IMG_W = 416
NUM_CLASSES = 4


""" Model """
model_path = os.path.join("files","model.h5")
print(model_path)
model = tf.keras.models.load_model(model_path)
# model.summary()


def get_colormap():
    colormap = [[0,0,0],[0,0,128],[0,128,0],[128,0,0]]
    classes = [
        "Background",
        "road",
        'nonroad'
        "ignore",
    ]
    return classes, colormap

""" Colormap """
CLASSES, COLORMAP = get_colormap()

def test_load_dataset(path):

    test_x = sorted(glob(os.path.join(path, "images","*")))[:160]
    return test_x


def grayscale_to_rgb(mask, classes, colormap):
    h, w, _ = mask.shape
    mask = mask.astype(np.int32)
    output = []

    for i, pixel in enumerate(mask.flatten()):
        output.append(colormap[pixel])
    print(h)
    output = np.reshape(output, (h, w, 3))
    return output

def save_results(image, pred):
    # h, w, _ = image.shape
    # line = np.ones((h, 10, 3)) * 255

    pred = np.expand_dims(pred, axis=-1)
    pred = grayscale_to_rgb(pred, CLASSES, COLORMAP)
    return pred


def road_extraction(frame,frame_width,frame_height):
    """ Reading the image """
    image_y = frame.copy()
    image = cv2.resize(frame, (IMG_W, IMG_H))
    
    image = image/255.0
    image = np.expand_dims(image, axis=0)

    """ Prediction """
    pred = model.predict(image, verbose=0)[0]
    pred = np.argmax(pred, axis=-1)
    pred = pred.astype(np.float32)

    """ Saving the prediction """
    final_frame=save_results(image_y, pred)
    final_frame = (final_frame * 255).astype(np.uint8)

    final_frame=cv2.resize(final_frame,(frame_width,frame_height))


    overlay = cv2.addWeighted(image_y, 1, final_frame, 0.5, 0)
    roi_frame=contour(image_y,final_frame)

    return overlay,roi_frame
    