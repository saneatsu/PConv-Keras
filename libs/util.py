import os
import sys 
from random import randint
import itertools
import numpy as np
import cv2
from PIL import Image
sys.path.append(os.pardir)

import const as cst


def random_mask(height, width, channels=3):
    """Generates a random irregular mask with lines, circles and elipses"""    
    img = np.zeros((height, width, channels), np.uint8)
    
    line_size = int((width + height) * 0.01)
    circle_size = int((width + height) * 0.023)
    
    if width < 64 or height < 64:
        raise Exception("Width and Height of mask must be at least 64!")

    # Draw random lines
#     for _ in range(randint(1, 10)): # randint(1, 20)
#         x1, x2 = randint(1, width), randint(1, width)
#         y1, y2 = randint(1, height), randint(1, height)
#         thickness = randint(3, size)
#         cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)
                
    # Draw random ellipses
    for _ in range(randint(1, 10)):
        x1, y1 = randint(1, width), randint(1, height)
        s1, s2 = randint(1, width), randint(1, height)
        a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
        thickness = randint(3, line_size)
        cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)
    
    # Draw random circles
    for _ in range(randint(1, 20)):
        x1, y1 = randint(1, width), randint(1, height)
        radius = randint(3, circle_size)
        cv2.circle(img,(x1,y1),radius,(1,1,1), -1)

    return 1-img