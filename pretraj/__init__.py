import numpy as np
import torch
import json
import random
from pretraj import vehicle

torch.manual_seed(3)
np.random.seed(3)
random.seed(3)

# number of frames for each car pairs
NUM_FRAMES = 200
NUM_PAIRS = 500

# file paths related to ngsim
NGSIM_DIR = 'pretraj/ngsim/'
NGSIM_PATH = NGSIM_DIR + 'NGSIM.csv'
TEMP_REDUCED_NGSIM_PATH = NGSIM_DIR + 'TEMP_REDUCED_NGSIM.csv'
REDUCED_NGSIM_PATH = NGSIM_DIR + 'REDUCED_NGSIM.csv'
REDUCED_NGSIM_JSON_PATH = NGSIM_DIR + 'REDUCED_NGSIM.json'

# load data to python data structure
with open(REDUCED_NGSIM_JSON_PATH) as fp:
    pairs_info = json.load(fp)

vehicle_pairs_list = \
  [(vehicle.Vehicle(**pairs_info[i]['ego']),
    vehicle.Vehicle(**pairs_info[i]['pre']))
    for i in range(NUM_PAIRS)]

# outliers
outliers = [113]

# result paths
RESULT_DIR = 'pretraj/res/'
FIXED_OBSERVE_JSON_PATH = RESULT_DIR + 'fixed_observe.json'
FIXED_OBSERVE_FIG_PATH = RESULT_DIR + 'fixed_observe.png'
FIXED_PREDICT_JSON_PATH = RESULT_DIR + 'fixed_predict.json'
FIXED_PREDICT_FIG_PATH = RESULT_DIR + 'fixed_predict.png'
RUNTIME_PATH = RESULT_DIR + 'runtime.json'

# checkpoint
CHECKPOINT_DIR = 'pretraj/checkpoint/'
PRETRAIN_MODEL_PATH = CHECKPOINT_DIR + 'pretrain_model.pt'
OFFLINE_REGRESSION = CHECKPOINT_DIR + 'offline_regression.npy'