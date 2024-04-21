
"Some global configure settings here"
import os

INPUT_DATA_DIR = os.path.join(os.path.dirname(__file__), "Data\\test_h5_folder")
INPUT_VALID_DATA_DIR = os.path.join(os.path.dirname(__file__), "Data\\test_h5_folder")
INPUT_ANOTATION_DIR = ""
OUTPUT_DIR = ""
LOG_FILE_PATH = ""
TRAIN_RATIO = 1  # ow many percentage of data we are using to train"
BATCH_SIZE = 4
SLICES = [17,18,19]
CROP_SIZE =(20,20)
MAX_FILE_LIMIT = 1000 # how many files we want to process
ACCELERATE_RATE = 4