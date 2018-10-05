import os
import sys
import pathlib
import cv2
import numpy as np
import json

sys.path.append(os.pardir) # for using const.py

import const as cst

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

img_cnt      = 0


def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8), -1)
    return cv_img


# Make bmp file to RESIZED_PATH from ORIGINAL_PATH
for i in range(len(cst.ORIGINAL_PATH)):
    d = cst.ORIGINAL_PATH[i]
    d_child=[os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
    print(cst.ORIGINAL_PATH[i])
    print(d_child)

    for j in range(len(d_child)):
        print(d_child[j])

        if 'en' in d_child[j]:
            print('Generate')
            onlyfiles = [filepath.absolute() for filepath in pathlib.Path(d_child[j]).glob('**/*')]

            for k in range(len(onlyfiles)):
                if(str(onlyfiles[k]).endswith('.jpg')  or
                   str(onlyfiles[k]).endswith('.jpeg') or
                   str(onlyfiles[k]).endswith('.png')  or
                   str(onlyfiles[k]).endswith('.bmp')):
                    pass
                else:
                    print('skip this file:' + str(onlyfiles[k]))
                    continue

                img = cv2.imread(str(onlyfiles[k]).replace('\\', '/'))
                height, width, channels = img.shape
                
                if(height < cst.MAX_HEIGHT or width < cst.MAX_WIDTH):
                    print('skip this image:' + str(onlyfiles[k]))
                elif cst.MAX_HEIGHT < height or cst.MAX_WIDTH < width:
                    # Get scaling factor
                    scaling_factor = cst.MAX_HEIGHT / float(height)
                    # 216/2688 = 0.09523809523
                    # 512/5376 = 0.09523809523
                    if cst.MAX_WIDTH / float(width) > scaling_factor: # Basically don't pass through here
                        scaling_factor = cst.MAX_WIDTH / float(width)

                    # Resize image(reduction)
                    img = cv2.resize(img,
                                     None,
                                     fx=scaling_factor,
                                     fy=scaling_factor,
                                     interpolation=cv2.INTER_AREA)
                    sh_height, sh_width, sh_channels = img.shape

                    if(sh_height > cst.MAX_HEIGHT):
                        crop_height = int(sh_height/2)
                        crop_img = img[crop_height-128:crop_height+128, 0:512]
                    elif(sh_width > cst.MAX_WIDTH):
                        crop_width = int(sh_height/2)
                        crop_img = img[0:256, crop_width-128:crop_width+128+256]
                    else:
                        crop_img = img # Basically pass through here

                        
                    params = list()
                    params.append(cv2.IMWRITE_PNG_COMPRESSION)
                    params.append(8)
                    img_cnt += 1
                    img_file_name = cst.RESIZED_PATH[i] + \
                                    '/img_' + \
                                    '{0:07d}'.format(img_cnt) + \
                                    '.bmp'
                    cv2.imwrite(img_file_name, crop_img, params)
