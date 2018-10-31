# NOTE: You need to check directory's permission

MAX_HEIGHT = 1536 # 1536 # 1280 # 1024 # 2048 # 2688
MAX_WIDTH  = 3072 # 3072 # 2560 # 2048 # 4096 # 5376

CROP_HEIGHT = 512 # 256, 512
CROP_WIDTH  = 1024 # 512, 1024

MNT_PATH   = '/nfs/host/PConv-Keras'

# house-dataset-src dir
ORIGINAL_PATH = ['{}/house-dataset-src/original/'.format(MNT_PATH)]
RESIZED_PATH  = ['{}/house-dataset-src/resized-2688-5376/'.format(MNT_PATH)]

# house-dataset dir
TRAIN_PATH    = '/mnt/PConv-Keras/house-dataset/train-crop-256-512'
VAL_PATH      = '/mnt/PConv-Keras/house-dataset/valid-crop-256-512'
TEST_PATH     = '/mnt/PConv-Keras/house-dataset/test-crop-256-512'

# data dir
WEIGHT_PATH   = '{}/data/model/weight-crop-512-1024/'.format(MNT_PATH)
TFLOG_PATH    = '{}/data/model/tflogs'.format(MNT_PATH)
ERRLOG_PATH   = '{}/error_log/'.format(MNT_PATH)