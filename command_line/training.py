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
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras import backend as K
import keras

sys.path.append(os.pardir)

import const as cst
from libs.pconv_model import PConvUnet
from libs.util import random_mask

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
BATCH_SIZE = 7 # ResourceExhaustedError
plt.ioff()

print(BATCH_SIZE)
print(cst.TRAIN_PATH)
print(cst.VAL_PATH)
print(cst.WEIGHT_PATH)
print(cst.TFLOG_PATH)

class DataGenerator(ImageDataGenerator):
    def __init__(self, random_crop_size=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert random_crop_size == None or len(random_crop_size) == 2
        self.random_crop_size = random_crop_size
        
    def has_many_mask(self, mask):
        height, width = mask.shape[0], mask.shape[1]
        masked_pixels = []

        for y in range(height):
            for x in range(width):
                if mask[y, x, 0] == 0: # 0: black
                    masked_pixels.append(mask[y, x, 0])                    
                    
        mask_rate = len(masked_pixels)/262144*100
        print("")
        print(str(len(masked_pixels)/262144*100)) # 512x512=262,144

        if mask_rate < 1:
            return False
        
        return True

        # if len(masked_pixels) != 0:
        #    return True
        # return False
            
    def random_crop(self, ori):
        assert ori.shape[3] == 3
        if ori.shape[1] < self.random_crop_size[0] or ori.shape[2] < self.random_crop_size[1]:
            raise ValueError(f"Invalid random_crop_size : original = {ori_img.shape}, crop_size = {self.random_crop_size}")

        height, width = ori.shape[1], ori.shape[2]
        dy, dx = self.random_crop_size
                    
        # while True:
        x = np.random.randint(0, width - dx - 1)
        y = np.random.randint(0, height - dy - 1)

        # Check ratio of mask
        croped_ori = ori[:, y:(y+dy), x:(x+dx), :]
        # croped_mask = mask[:, y:(y+dy), x:(x+dx), :]

        # if self.has_many_mask(croped_mask[0]):
        #    break
            
        return croped_ori
    
    def save_img(cnt, mask=None, masked=None, croped_ori=None, croped_mask=None):
        save_dir = '/nfs/host/PConv-Keras/sample_images/'
        
        if mask:
            save_mask = Image.fromarray(np.uint8((mask[0,:,:,:] * 1.)*255))
            save_mask.save("{}_mask.jpg".format(save_dir, cnt))
        if masked:
            save_masked = Image.fromarray(np.uint8((masked[0,:,:,:] * 1.)*255))
            save_masked.save("{}_masked.jpg".format(save_dir, cnt))
        if croped_ori:
            save_croped_ori = Image.fromarray(np.uint8((croped_ori[0,:,:,:] * 1.)*255))
            save_croped_ori.save("{}_croped_ori.jpg".format(save_dir, cnt))                    
        if croped_mask:
            save_croped_masked = Image.fromarray(np.uint8((croped_mask[0,:,:,:] * 1.)*255))
            save_croped_masked.save("{}_save_croped_masked.jpg".format(save_dir, cnt))
            
        print('Save')

    def flow_from_directory(self, directory, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)
        
        cnt=0

        # Data augmentation
        while True:                   
            # Get augmented image samples
            ori = next(generator)
            ori_length = ori.shape[0]

            # Crop ori images
            croped_ori = self.random_crop(ori)
            croped_ori_length = croped_ori.shape[0]

            # Get masks for each image sample
            mask = np.stack([random_mask(croped_ori.shape[1], croped_ori.shape[2]) for _ in range(croped_ori_length)], axis=0)           

            # Crop ori, mask and masked images
            # croped_ori, croped_mask = self.random_crop(ori, mask)

            # Apply masks to all image sample
            masked = deepcopy(croped_ori)
            masked[mask == 0] = 1

            # Yield ([ori, masl],  ori) training batches
            gc.collect()        
            
#             self.has_many_mask(mask[0,:,:,:])

            # self.save_img(cnt=cnt, mask=mask, masked=masked) 
            
            cnt += 1

            yield [masked, mask], croped_ori

print(cst.MAX_HEIGHT)
print(cst.MAX_WIDTH)            
print(cst.CROP_HEIGHT)
print(cst.CROP_WIDTH)

train_datagen = DataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    horizontal_flip=True,
    random_crop_size=(cst.CROP_HEIGHT, cst.CROP_WIDTH)
)
train_generator = train_datagen.flow_from_directory(
    cst.TRAIN_PATH,
    target_size=(cst.MAX_HEIGHT, cst.MAX_WIDTH),
    batch_size=BATCH_SIZE
)
    
val_datagen = DataGenerator(
    rescale=1./255,
    random_crop_size=(cst.CROP_HEIGHT, cst.CROP_WIDTH)
)
val_generator = val_datagen.flow_from_directory(
    cst.VAL_PATH,
    target_size=(cst.MAX_HEIGHT, cst.MAX_WIDTH),
    batch_size=BATCH_SIZE,
    seed=1
)

model = PConvUnet(weight_filepath=cst.WEIGHT_PATH)

# model.load_weights('/nfs/host/PConv-Keras/data/model/resize-1535x3072/weight_256x256_GPU-4_Batch-27_Mask-Circle/60_weights_2019-01-04-07-06-58.h5')

model.fit(
    train_generator,
    steps_per_epoch=8000//BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=8000//BATCH_SIZE,
    epochs=300,
    plot_callback=None,
    callbacks=[
        TensorBoard(log_dir=cst.TFLOG_PATH, write_graph=False),
    ])

# $ tensorboard --logdir=/nfs/host/PConv-Keras/data/model/tflogs --port 8082
