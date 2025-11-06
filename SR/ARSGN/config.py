# config.py
""" The Adam optimizer is employed for
optimization with a mini-batch size of 16, beta1 = 0.9, beta2 =
0.999, and epsilon = 1e - 8. The learning rate is initialized to
1e - 4 and decreases by a factor of 10 for every 500 epochs.
The original images are downsampled with the bicubic
function. """

import os
import torch

# Parametri di I/O
DATA_DIR_LR = "C:/data_tif/lr"
DATA_DIR_HR = "C:/data_tif/hr"
WEIGHTS_DIR = "SR\ARSGN\weights_new"
SAVE_FREQUENCY = 10

# Parametri di Training
HR_PATCH_SIZE = 16
BATCH_SIZE = 5
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4

# Parametri del Modello (se non passati tramite argparse)
SCALE_FACTOR = 4
N_FEATS = 16
ALPHA_FFL = 0.1

# Parametri dell'ottimizzazione

ADAM_BETA1 =  0.9 
ADAM_BETA2 = 0.999 
ADAM_EPSILON =  1e-8 

# schedular parameters

LR_DECAY_STEP = 500
LR_DECAY_FACTOR = 0.1

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output test
OUTPUT_DIR = "SR/ARSGN/SR_images"
DATA_DIR_TEST_LR = "SR/ARSGN/data/lr_test_64"
DATA_DIR_TEST_HR = "SR/ARSGN/data/hr_test"