import os
import sys
import gc
import datetime
import cv2
from copy import deepcopy
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras import backend as K

sys.path.append(os.pardir)
sys.path.append(cst.MNT_PATH)

import const as cst
from libs.pconv_model import PConvUnet
from libs.util import random_mask

os.environ["CUDA_VISIBLE_DEVICES"]="0"

TRAIN_DIR = '{}/house-dataset/train'.format(cst.MNT_PATH)
VAL_DIR   = '{}/house-dataset/valid'.format(cst.MNT_PATH)
TEST_DIR  = '{}/house-dataset/test'.format(cst.MNT_PATH)
BATCH_SIZE = 16
plt.ioff()

class DataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
        while True:
            # Get augmented image samples
            ori = next(generator)

            # Get masks for each image sample
            mask = np.stack([random_mask(ori.shape[1], ori.shape[2]) for _ in range(ori.shape[0])], axis=0)

            # Apply masks to all image sample
            masked = deepcopy(ori)
            masked[mask == 0] = 1

            # Yield ([ori, masl],  ori) training batches
            # print(masked.shape, ori.shape)
            gc.collect()
            yield [masked, mask], ori


# Create training generator
train_datagen = DataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(256, 256), batch_size=BATCH_SIZE
)

# Create validation generator
val_datagen = DataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(256, 256), batch_size=BATCH_SIZE, seed=1
)

# Instantiate the model
model = PConvUnet(weight_filepath="{}/data/model/".format(cst.MNT_PATH))
# Run training for certain amount of epochs
model.fit(
    train_generator,
    steps_per_epoch=10, # Fix
    validation_data=val_generator,
    validation_steps=100,
    epochs=3000, # Fix
    plot_callback=None,
    callbacks=[
        TensorBoard(log_dir='{}/data/model/initial_training'.format(cst.MNT_PATH), write_graph=False)
    ]
)
