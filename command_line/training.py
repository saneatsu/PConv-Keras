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

import const as cst
from libs.pconv_model import PConvUnet
from libs.util import random_mask

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

BATCH_SIZE = 8 # ResourceExhaustedError
plt.ioff()

class DataGenerator(ImageDataGenerator):      
    def __init__(self, random_crop_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
                
        assert random_crop_size == None or len(random_crop_size) == 2
        self.random_crop_size = random_crop_size

            
    def random_crop(self, ori_img):
        assert ori_img.shape[3] == 3
        if ori_img.shape[1] < self.random_crop_size[0] or ori_img.shape[2] < self.random_crop_size[1]:
            raise ValueError(f"Invalid random_crop_size : original = {ori_img.shape}, crop_size = {self.random_crop_size}")

        height, width = ori_img.shape[1], ori_img.shape[2]
        dy, dx = self.random_crop_size        
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        crop_img = ori_img[y:(y+dy), x:(x+dx), :]
        
        return crop_img
    
    
    def flow_from_directory(self, directory, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
        
        # Data augmentation
        while True:
            # Get augmented image samples
            ori = next(generator)
            ori_length = ori.shape[0]            
            crop_ori = self.random_crop(ori) 

            # Get masks for each image sample
            mask = np.stack([random_mask(crop_ori.shape[1], crop_ori.shape[2]) for _ in range(ori_length)], axis=0)

            # Apply masks to all image sample
            masked = deepcopy(crop_ori)
            masked[mask == 0] = 1

            # Yield ([ori, masl],  ori) training batches
            gc.collect()          
                        
            yield [masked, mask], crop_ori
            

train_datagen = DataGenerator(
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    rescale=1./255,
                    horizontal_flip=True,
                    random_crop_size=(cst.CROP_HEIGHT, cst.CROP_WIDTH))
train_generator = train_datagen.flow_from_directory(
                    cst.TRAIN_PATH,
                    target_size=(cst.CROP_HEIGHT, cst.CROP_WIDTH),
                    batch_size=BATCH_SIZE)


val_datagen = DataGenerator(
                    rescale=1./255,
                    random_crop_size=(cst.CROP_HEIGHT, cst.CROP_WIDTH))
val_generator = val_datagen.flow_from_directory(
                    cst.VAL_PATH,
                    target_size=(cst.CROP_HEIGHT, cst.CROP_WIDTH),
                    batch_size=BATCH_SIZE,
                    seed=1)


model = PConvUnet(weight_filepath=cst.WEIGHT_PATH)
model.fit(
    train_generator,
    steps_per_epoch=100,
    validation_data=val_generator,
    validation_steps=500,
    epochs=100,
    plot_callback=None,
    callbacks=[
        TensorBoard(log_dir=cst.TFLOG_PATH, write_graph=False)
    ])
        
# $ tensorboard --logdir=/nfs/host/PConv-Keras/data/model/tflogs --port 8082
