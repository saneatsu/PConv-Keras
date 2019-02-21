# NOTE: You need to check directory's permission

MAX_HEIGHT = 1536 # 1536 # 768 # 1280 # 1024 # 2048 # 2688
MAX_WIDTH  = 3072 # 3072 # 1536 # 2560 # 2048 # 4096 # 5376

CROP_HEIGHT   = 512 # 256, 384, 512
CROP_WIDTH    = 512 # 512, 768, 1024

RESIZE_HEIGHT = 1536
RESIZE_WIDTH  = 3072

MNT_PATH   = '/nfs/host/PConv-Keras'

# house-dataset-src dir
ORIGINAL_PATH = ['{}/house-dataset-src/original/'.format(MNT_PATH)] # 9317 imag bves
RESIZED_PATH  = ['{}/house-dataset/resize-train-1536x3072/00'.format(MNT_PATH)]

TRAIN_PATH    = '{}/house-dataset/resize-train-768x1536'.format(MNT_PATH)
VAL_PATH      = '{}/house-dataset/resize-valid-768x1536'.format(MNT_PATH)
TEST_PATH     = '{}/house-dataset/test-crop-256-512'.format(MNT_PATH)

# data dir
WEIGHT_PATH   = '{}/data/model/resize-1536x3072/'.format(MNT_PATH)
TFLOG_PATH    = '{}/data/model/resize-1536x3072/'.format(MNT_PATH)
ERRLOG_PATH   = '{}/error_log/'.format(MNT_PATH)


"""
# 128x256
weight-crop-128x256
tflogs-crop-128x256

# 256x512
{}/data/model/weight-resize-1536x3072/weight-256x512

"""

"""
saneatsu@X99GPU /n/h/PConv-Keras> ls house-dataset -l
total 2016

#
# Original images
#
drwxrwxrwx 1 root root       0 Oct 18 12:18 ori-train-5376x2688x7922/
drwxrwxrwx 1 root root       0 Oct 18 12:19 ori-valid-5376x2688x844/
drwxrwxrwx 1 root root       0 Oct 18 12:18 ori-test-5376x2688x3/

#
# Resized images
#
drwxrwxrwx 1 root root       0 Oct  9 17:26 resize-test-256x512x3/
drwxrwxrwx 1 root root       0 Oct  9 17:26 resize-train-256x512x5/
drwxrwxrwx 1 root root       0 Oct  9 17:26 resize-valid-256x512x5/

drwxrwxrwx 1 root root 1388544 Oct  9 12:31 resize-train-512x1024x7999/
drwxrwxrwx 1 root root  335872 Oct  9 12:29 resize-test-512x1024x1999/
drwxrwxrwx 1 root root  339968 Oct  9 12:33 resize-valid-512x1024x1999/

"""
