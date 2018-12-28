import os
import sys 
from random import randint
import itertools
import numpy as np
import cv2
from PIL import Image
sys.path.append(os.pardir)

import const as cst


# def random_mask(height, width, channels=3):
#     """Generates a random irregular mask with lines, circles and elipses"""    
#     img = np.zeros((height, width, channels), np.uint8)

#     # Set size scale
#     # size = int((width + height) * 0.03) # 4608*0.03=138
#     size = int((width + height) * 0.01)
#     if width < 64 or height < 64:
#         raise Exception("Width and Height of mask must be at least 64!")
    
#     # Draw random lines
#     for _ in range(randint(30, 50)): # randint(1, 20)
#         x1, x2 = randint(1, width), randint(1, width)
#         y1, y2 = randint(1, height), randint(1, height)
#         thickness = randint(3, size)
#         cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)
        
#     # Draw random circles
#     for _ in range(randint(80, 100)): # randint(1, 20)
#         x1, y1 = randint(1, width), randint(1, height)
#         radius = randint(3, size)
#         cv2.circle(img,(x1,y1),radius,(1,1,1), -1)
        
#     # Draw random ellipses
#     for _ in range(randint(80, 100)): # randint(1, 20)
#         x1, y1 = randint(1, width), randint(1, height)
#         s1, s2 = randint(1, width), randint(1, height)
#         a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
#         thickness = randint(3, size)
#         cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)
    
#     return 1-img

def print_mask_rate(mask):
        height, width = mask.shape[0], mask.shape[1]
        masked_pixels = []

        for y in range(height):
            for x in range(width):
                if mask[y, x, 0] == 0: # 0: black
                    masked_pixels.append(mask[y, x, 0])
        
#         print("Rate: " + str(len(masked_pixels)*3/cst.CROP_HEIGHT*cst.CROP_WIDTH))

def random_mask(height, width, channels=3):
    """Generates a random irregular mask with lines, circles and elipses"""    
    margin = 30
    min_x = margin
    min_y = margin
    max_x = cst.CROP_WIDTH - margin
    max_y = cst.CROP_HEIGHT - margin    
        
    img_height = height-margin*2
    img_width = width-margin*2
    img = np.zeros((img_height, img_width, channels), np.uint8)    
        
    circle_size = int((width + height) * 0.07)
    line_size = int((width + height) * 0.04)
    if width < 20 or height < 20:
        raise Exception("Width and Height of mask must be at least 64!")
    
    # Draw random lines
    for _ in range(randint(0, 2)):
        x1, x2 = randint(1, img_width), randint(1, img_width)
        y1, y2 = randint(1, img_height), randint(1, img_height)
        thickness = randint(3, line_size)
        cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)
    
    # Draw random ellipses
    for _ in range(randint(0, 2)):
        x1, y1 = randint(1, img_width), randint(1, img_height)
        s1, s2 = randint(1, img_width), randint(1, img_height)
        a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
        thickness = randint(3, line_size)
        cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)
        
    # Draw random circles
    for _ in range(randint(1, 4)):
        x1, y1 = randint(1, img_width), randint(1, img_height)
        radius = randint(3, circle_size)
        cv2.circle(img,(x1,y1),radius,(1,1,1), -1)        
    
    blank_img = np.zeros((height, width, channels), np.uint8)
    blank_img[margin:img.shape[0]+margin, margin:img.shape[1]+margin] = img    
    
#     print_mask_rate(blank_img)    
#     save_img = Image.fromarray(np.uint8((img[:,:,:] * 1.)*255))
#     save_img.save("/nfs/host/PConv-Keras/sample_images/save_img.jpg")
#     save_blank_img = Image.fromarray(np.uint8((blank_img[:,:,:] * 1.)*255))
#     save_blank_img.save("/nfs/host/PConv-Keras/sample_images/save_blank_img.jpg")
    
    
    return 1-blank_img
