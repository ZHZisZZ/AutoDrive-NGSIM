import numpy as np
import torch
import random

torch.manual_seed(3)
np.random.seed(3)
random.seed(3)

# number of frames for each car pairs
NUM_FRAMES = 200

# file paths related to ngsim
NGSIM_DIRT = 'pretraj/ngsim/'
NGSIM_PATH = NGSIM_DIRT + 'NGSIM.csv'
TEMP_REDUCED_NGSIM_PATH = NGSIM_DIRT + 'TEMP_REDUCED_NGSIM.csv'
REDUCED_NGSIM_PATH = NGSIM_DIRT + 'REDUCED_NGSIM.csv'
REDUCED_NGSIM_JSON_PATH = NGSIM_DIRT + 'REDUCED_NGSIM.json'

# outliers
outliers = [113]

# result paths
RESULT_DIR = 'pretraj/res/'
FIXED_OBSERVE_JSON_PATH = RESULT_DIR + 'fixed_observe.json'
FIXED_OBSERVE_FIG_PATH = RESULT_DIR + 'fixed_observe.png'
FIXED_PREDICT_JSON_PATH = RESULT_DIR + 'fixed_predict.json'
FIXED_PREDICT_FIG_PATH = RESULT_DIR + 'fixed_predict.png'
RUNTIME_PATH = RESULT_DIR + 'runtime.json'