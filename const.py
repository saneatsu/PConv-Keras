# NOTE: You need to check directory's permission
MAX_HEIGHT = 1536 # 1536 # 768 # 1280 # 1024 # 2048 # 2688
MAX_WIDTH  = 3072 # 3072 # 1536 # 2560 # 2048 # 4096 # 5376

CROP_HEIGHT   = 512 # 256, 384, 512
CROP_WIDTH    = 512 # 512, 768, 1024

RESIZE_HEIGHT = 1536
RESIZE_WIDTH  = 3072

# MNT_PATH = '/nfs/host/PConv-Keras' # For Spacely_Multi_GPU
MNT_PATH = '/nfs/host/PConv-Keras' # For Spacly_Lab

# house-dataset-src dir
ORIGINAL_PATH = ['{}/house-dataset-src/original/'.format(MNT_PATH)] # 9317 images
RESIZED_PATH  = ['{}/house-dataset/resize-train-1536x3072/00'.format(MNT_PATH)]

TRAIN_PATH    = '{}/house-dataset/resize-train-768x1536'.format(MNT_PATH)
VAL_PATH      = '{}/house-dataset/resize-valid-768x1536'.format(MNT_PATH)
TEST_PATH     = '{}/house-dataset/test-crop-256-512'.format(MNT_PATH)

# For mix training
TRAIN_SMALL_SIZE = '{}/house-dataset/resize-train-512x1024'.format(MNT_PATH)
TRAIN_MEDIUM_SIZE = '{}/house-dataset/resize-train-768x1536'.format(MNT_PATH)
TRAIN_LARGE_SIZE = '{}/house-dataset/resize-train-1536x3072'.format(MNT_PATH)
VALID_SMALL_SIZE = '{}/house-dataset/resize-valid-512x1024'.format(MNT_PATH)
VALID_MEDIUM_SIZE = '{}/house-dataset/resize-valid-768x1536'.format(MNT_PATH)
VALID_LARGE_SIZE = '{}/house-dataset/resize-valid-1536x3072'.format(MNT_PATH)     

# Save dir
WEIGHT_PATH   = '{}/data/model/resize-1536x3072/512x512_GPU-2_Batch-4_NewModel/weight/'.format(MNT_PATH)
TFLOG_PATH    = '{}/data/model/resize-1536x3072/512x512_GPU-2_Batch-4_NewModel/tflogs'.format(MNT_PATH)
ERRLOG_PATH   = '{}/error_log/'.format(MNT_PATH)

