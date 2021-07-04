import json
import numpy as np
from pretraj.vehicle import Vehicle
from pretraj import predict
from pretraj.merics import ADE, FDE
from pretraj import *


with open(REDUCED_NGSIM_JSON_PATH) as fp:
  pair_info = json.load(fp)

number_vehicles = len(pair_info)
observe_frames = 30
predict_frames = 18

metrics = ADE

error_IDM = 0; hard_braking_IDM = 0
error_adapt = 0; hard_braking_adpat = 0


for i in range(number_vehicles):
  ego = Vehicle(**pair_info[i]['ego'])
  pre = Vehicle(**pair_info[i]['pre'])

  groundtruth_record = ego.space_headway_vector[observe_frames:observe_frames+predict_frames]

  ds_record, _, _, hard_braking = predict.predict(ego, pre, observe_frames, predict_frames, 'IDM')
  error_IDM += ADE(np.array(ds_record), np.array(groundtruth_record))
  hard_braking_IDM += hard_braking

  ds_record, _, _, hard_braking = predict.predict(ego, pre, observe_frames, predict_frames, 'adapt')
  error_adapt += ADE(np.array(ds_record), np.array(groundtruth_record))
  hard_braking_adpat += hard_braking

error_IDM /= number_vehicles
error_adapt /= number_vehicles
print('error of fixed IDM model:', error_IDM)
print('hardbraking of fixed IDM model:', hard_braking_IDM)
print('error of adaptation model:', error_adapt)
print('hardbraking of adaptation model:', hard_braking_adpat)