
"Some global configure settings here"
import os

INPUT_DATA_DIR = os.path.join(os.path.dirname(__file__), "D:\Repos\CS7643\project\singlecoil_train")
INPUT_VALID_DATA_DIR = os.path.join(os.path.dirname(__file__), "D:\Repos\CS7643\project\knee_singlecoil_val\singlecoil_val")
INPUT_ANOTATION_DIR = ""
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "\saved_models")
LOG_FILE_PATH = ""
TRAIN_RATIO = 1  # ow many percentage of data we are using to train"
BATCH_SIZE = 4
SLICES = [17,18,19]
CROP_SIZE =(20,20)
MAX_FILE_LIMIT = 1000 # how many files we want to process
ACCELERATE_RATE = 4
NUM_CHANNEL_FIRST_LAYER_OUTPUT = 32 # number of output channels of the first layer of the model
NUM_POOL_LAYERS = 4
DROPOUT_PROB = 0.0
IS_SSIM_LOSS = True
THRESHOLD_SSIM = 0.9
THRESHOLD_MSE = 0.1
EPOCHS = 5
