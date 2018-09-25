import os
import pathlib
import cv2
import numpy as np
import json

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Make bmp file to target folder from src folder
# NOTE: You need to check tgt_training_folder's permission
src_folder          = ['/mnt/PConv-Keras/house-dataset-src/original/']
tgt_training_folder = ['/mnt/PConv-Keras/house-dataset-src/resized/']

max_height = 256
max_width  = 256
img_cnt    = 0


def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img


for i in range(len(src_folder)):
    d = src_folder[i]
    d_child=[os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
    print(src_folder[i])
    print(d_child)

    for j in range(len(d_child)):
        print(d_child[j])
        
        if 'en' in d_child[j]:
            print('Generate')
            onlyfiles = [filepath.absolute() for filepath in pathlib.Path(d_child[j]).glob('**/*')]
            
            for k in range(len(onlyfiles)):
                if(str(onlyfiles[k]).endswith('.jpg') or str(onlyfiles[k]).endswith('.jpeg') or
                  str(onlyfiles[k]).endswith('.png') or str(onlyfiles[k]).endswith('.bmp')):
                    pass
                else:
                    print("skip this file:" + str(onlyfiles[k]))
                    continue

                img = cv2.imread(str(onlyfiles[k]).replace('\\', '/'))
                height, width, channels = img.shape

                if(height < max_height or width < max_width):
                    print('skip this image:'+str(onlyfiles[k]))
                elif max_height < height or max_width < width:
                    # get scaling factor
                    scaling_factor = max_height / float(height)
                    if max_width / float(width) > scaling_factor:
                        scaling_factor = max_width / float(width)

                    # resize image(reduction)
                    img = cv2.resize(img,
                                     None,
                                     fx=scaling_factor,
                                     fy=scaling_factor,
                                     interpolation=cv2.INTER_AREA)
                    sh_height, sh_width, sh_channels = img.shape

                    if(sh_height>256):
                        crop_img = img[int(sh_height/2)-128:int(sh_height/2)+128, 0:256]
                    elif(sh_width>256):
                        crop_img = img[0:256,int(sh_height/2)-128:int(sh_height/2)+128]
                    else:
                        crop_img = img

                    params = list()
                    params.append(cv2.IMWRITE_PNG_COMPRESSION)
                    params.append(8)
                    img_cnt+=1
                    img_file_name=tgt_training_folder[i]+"/img_"+"{0:07d}".format(img_cnt)+".bmp"
                    # print(img_file_name)
                    cv2.imwrite(img_file_name, crop_img, params)
