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

print('=================================================================')
print('BATCH_SIZE       : ' + str(BATCH_SIZE))
print('MAX_HEIGHT       : ' + str(cst.MAX_HEIGHT))
print('MAX_WIDTH        : ' + str(cst.MAX_WIDTH))
print('CROP_HEIGHT      : ' + str(cst.CROP_HEIGHT))
print('CROP_WIDTH       : ' + str(cst.CROP_WIDTH))
print()
print('TRAIN_PATH       : ' + cst.TRAIN_PATH)
print('VAL_PATH         : ' + cst.VAL_PATH)
print()
print('TRAIN_SMALL_SIZE : ' + cst.TRAIN_SMALL_SIZE)
print('TRAIN_MEDIUM_SIZE: ' + cst.TRAIN_MEDIUM_SIZE)
print('TRAIN_LARGE_SIZE : ' + cst.TRAIN_LARGE_SIZE)
print('VALID_SMALL_SIZE : ' + cst.VALID_SMALL_SIZE)
print('VALID_MEDIUM_SIZE: ' + cst.VALID_MEDIUM_SIZE)
print('VALID_LARGE_SIZE : ' + cst.VALID_LARGE_SIZE)
print()
print('WEIGHT_PATH      : ' + cst.WEIGHT_PATH)
print('TFLOG_PATH       : ' + cst.TFLOG_PATH)
print('=================================================================')

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
                            
        mask_rate = len(masked_pixels)/262144*100 # 262,144=512x512
        # print('\n' + str(mask_rate))
        
        #if mask_rate != 0:
        #    with open('/nfs/host/PConv-Keras/sample_images/mask_rate.txt', 'a') as f:
        #        f.write(str(mask_rate) + '\n')

        # if mask_rate < 1:
        #    return False        
        # return True

        if len(masked_pixels) != 0:
            return True

        return False
            
    def random_crop(self, ori, mask):
        assert ori.shape[3] == 3
        # FIXME: NOT COMMENT OUT 
        if ori.shape[1] < self.random_crop_size[0] or ori.shape[2] < self.random_crop_size[1]:
            raise ValueError(f"Invalid random_crop_size : original = {ori_img.shape}, crop_size = {self.random_crop_size}")

        height, width = ori.shape[1], ori.shape[2]
        dy, dx = self.random_crop_size

        x_max = width - dx
        y_max = height - dy
        
        while True:
            x = np.random.randint(0, x_max) if x_max > 0 else 0
            y = np.random.randint(0, y_max) if y_max > 0 else 0

            # Check ratio of mask
            croped_ori = ori[:, y:(y+dy), x:(x+dx), :]
            croped_mask = mask[:, y:(y+dy), x:(x+dx), :]        

            if self.has_many_mask(croped_mask[0]):
                break
                                 
        return croped_ori, croped_mask    
    
    
    def save_img(self, cnt, img_size_type, mask=None, masked=None, croped_ori=None, croped_mask=None, ori=None):
        save_dir = '/nfs/host/PConv-Keras/sample_images'
        
        if mask is not None:
            save_mask = Image.fromarray(np.uint8((mask[0,:,:,:] * 1.)*255))
            save_mask.save("{}/{}_mask({}).jpg".format(save_dir, cnt, img_size_type))
        if masked is not None:
            save_masked = Image.fromarray(np.uint8((masked[0,:,:,:] * 1.)*255))
            save_masked.save("{}/{}_masked({}).jpg".format(save_dir, cnt, img_size_type))
        if croped_ori is not None:
            save_croped_ori = Image.fromarray(np.uint8((croped_ori[0,:,:,:] * 1.)*255))
            save_croped_ori.save("{}/{}_croped_ori({}).jpg".format(save_dir, cnt, img_size_type))                    
        if croped_mask is not None:
            save_croped_masked = Image.fromarray(np.uint8((croped_mask[0,:,:,:] * 1.)*255))
            save_croped_masked.save("{}/{}_croped_masked({}).jpg".format(save_dir, cnt, img_size_type))
        if ori is not None:
            save_ori = Image.fromarray(np.uint8((ori[0,:,:,:] * 1.)*255))
            save_ori.save("{}/{}_ori({}).jpg".format(save_dir, cnt, img_size_type))
            
        # print('Save')

    def flow_from_directory(self, directory, *args, **kwargs):
        generator = super().flow_from_directory(directory, class_mode=None, *args, **kwargs)

        cnt = 0

        # Data augmentation
        while True:
            # Get augmented image samples
            ori = next(generator)
            ori_length = ori.shape[0]

            # Get masks for each image sample
            mask = np.stack([random_mask(ori.shape[1], ori.shape[2]) for _ in range(ori_length)], axis=0)1

            # Crop ori, mask and masked images
            croped_ori, croped_mask = self.random_crop(ori, mask)

            # Apply masks to all image sample
            masked = deepcopy(croped_ori)
            masked[croped_mask == 0] = 1

            # Yield ([ori, masl],  ori) training batches
            gc.collect()
            
            # Save image
            # self.save_img(cnt=cnt, img_size_type='', masked=masked)
            # cnt += 1
            
            yield [masked, croped_mask], croped_ori

            
train_datagen = DataGenerator(
    rescale=1./255,
    random_crop_size=(cst.CROP_HEIGHT, cst.CROP_WIDTH))
train_generator = train_datagen.flow_from_directory(
    cst.TRAIN_PATH,
    target_size=(cst.MAX_HEIGHT, cst.MAX_WIDTH),
    batch_size=BATCH_SIZE)
    
val_datagen = DataGenerator(
    rescale=1./255,
    random_crop_size=(cst.CROP_HEIGHT, cst.CROP_WIDTH))
val_generator = val_datagen.flow_from_directory(
    cst.VAL_PATH,
    target_size=(cst.MAX_HEIGHT, cst.MAX_WIDTH),
    batch_size=BATCH_SIZE,
    seed=1)

model = PConvUnet(weight_filepath=cst.WEIGHT_PATH)

model.load_weights('/nfs/host/PConv-Keras/data/model/resize-1536x3072/512x512_GPU-2_Batch-04_RecursiveTraining/weight/32_weights_2019-01-10-10-16-11.h5')

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
