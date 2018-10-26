# NOTE: You need to check directory's permission

MAX_HEIGHT = 2688
MAX_WIDTH  = 5376

CROP_HEIGHT = 512 # 256, 512
CROP_WIDTH  = 1024 # 512, 1024

MNT_PATH   = '/nfs/host/PConv-Keras'

# house-dataset-src dir
ORIGINAL_PATH = ['{}/house-dataset-src/original/'.format(MNT_PATH)]
RESIZED_PATH  = ['{}/house-dataset-src/resized-2688-5376/'.format(MNT_PATH)]

# house-dataset dir
TRAIN_PATH    = '{}/house-dataset/train-crop-256-512'.format(MNT_PATH)
VAL_PATH      = '{}/house-dataset/valid-crop-256-512'.format(MNT_PATH)
TEST_PATH     = '{}/house-dataset/test-crop-256-512'.format(MNT_PATH)

# data dir
WEIGHT_PATH   = '{}/data/model/weight-crop-256-512/'.format(MNT_PATH)
TFLOG_PATH    = '{}/data/model/tflogs'.format(MNT_PATH)
ERRLOG_PATH   = '{}/error_log/'.format(MNT_PATH)