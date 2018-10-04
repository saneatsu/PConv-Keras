# NOTE: You need to check directory's permission

MNT_PATH      = '/nfs/host/PConv-Keras'

# house-dataset-src dir
ORIGINAL_PATH = ['{}/house-dataset-src/original/'.format(MNT_PATH)]
RESIZED_PATH  = ['{}/house-dataset-src/resized/'.format(MNT_PATH)]

# house-dataset dir
TRAIN_PATH    = '{}/house-dataset/train'.format(MNT_PATH)
VAL_PATH      = '{}/house-dataset/valid'.format(MNT_PATH)
TEST_PATH     = '{}/house-dataset/test'.format(MNT_PATH)

# data dir
WEIGHT_PATH   = '{}/data/model/weight'.format(MNT_PATH)
TFLOG_PATH    = '{}/data/model/tflogs'.format(MNT_PATH)