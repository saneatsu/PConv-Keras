import os
import sys
import gc
import datetime
import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import cv2

import matplotlib
matplotlib.use('Agg') # For using matplot in server
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from IPython.display import clear_output
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras import backend as K

sys.path.append(os.pardir)

import const as cst
from libs.pconv_model import PConvUnet
from libs.util import random_mask
# from training import DataGenerator # Add

os.environ['CUDA_VISIBLE_DEVICES']='0'

BATCH_SIZE = 4


# Create testing generator
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
            
            
test_datagen = DataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
                    cst.TEST_PATH,
                    target_size=(256, 256),
                    batch_size=BATCH_SIZE,
                    seed=1)

# Pick out an example
test_data = next(test_generator)
(masked, mask), ori = test_data

# Load weights from previous run
model = PConvUnet(weight_filepath='data/model/')
model.load(
    '{}/data/model/weight/3000_weights_2018-09-29-08-46-51.h5'.format(cst.MNT_PATH),
    train_bn=False,
    lr=0.00005)

n = 0
for (masked, mask), ori in tqdm(test_generator):    
    # Run predictions for this batch of images
    pred_img = model.predict([masked, mask])
    pred_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    # Clear current output and display test images
    for i in range(len(ori)):
        _, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(masked[i,:,:,:])
        axes[1].imshow(pred_img[i,:,:,:] * 1.)
        axes[0].set_title('Masked Image')
        axes[1].set_title('Predicted Image')
        axes[0].xaxis.set_major_formatter(NullFormatter())
        axes[0].yaxis.set_major_formatter(NullFormatter())
        axes[1].xaxis.set_major_formatter(NullFormatter())
        axes[1].yaxis.set_major_formatter(NullFormatter())
                
        plt.savefig('{}/data/output_samples/img_{}_{}.png'.format(cst.MNT_PATH, i, pred_time))
        plt.close()
        n += 1
        
    print(n)
    # Only create predictions for about 100 images
    # if n > 100:
    #     break
        
print('Finish')        