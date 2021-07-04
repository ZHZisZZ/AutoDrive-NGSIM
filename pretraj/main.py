import json
import numpy as np
from pretraj.vehicle import Vehicle
from pretraj import predict
from pretraj.merics import ADE, FDE
from pretraj import *


with open(REDUCED_NGSIM_JSON_PATH) as fp:
  pair_info = json.load(fp)

number_vehicles = 60
observe_frames = 30
predict_frames = 10

metrics = ADE

error_IDM = 0; hard_braking_IDM = 0
error_adapt = 0; hard_braking_adapt = 0
error_nn = 0; hard_braking_nn = 0

for i in range(number_vehicles):
  ego = Vehicle(**pair_info[i]['ego'])
  pre = Vehicle(**pair_info[i]['pre'])

  print(i)

  groundtruth_record = ego.space_headway_vector[observe_frames:observe_frames+predict_frames]

  ds_record, _, _, hard_braking = predict.predict(ego, pre, observe_frames, predict_frames, 'IDM')
  error_IDM += metrics(np.array(ds_record), np.array(groundtruth_record))
  hard_braking_IDM += hard_braking

  ds_record, _, _, hard_braking = predict.predict(ego, pre, observe_frames, predict_frames, 'adapt')
  error_adapt += metrics(np.array(ds_record), np.array(groundtruth_record))
  hard_braking_adapt += hard_braking

  ds_record, _, _, hard_braking = predict.predict(ego, pre, observe_frames, predict_frames, 'nn')
  error_nn += metrics(np.array(ds_record), np.array(groundtruth_record))
  hard_braking_nn += hard_braking


error_IDM /= number_vehicles
error_adapt /= number_vehicles
error_nn /= number_vehicles
print('error of fixed IDM model:', error_IDM)
print('hardbraking of fixed IDM model:', hard_braking_IDM)
print('error of adaptation model:', error_adapt)
print('hardbraking of adaptation model:', hard_braking_adapt)
print('error of nn model:', error_nn)
print('hardbraking of nn model:', hard_braking_nn)