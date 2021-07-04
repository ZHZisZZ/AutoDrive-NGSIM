import json
import numpy as np
from pretraj.vehicle import Vehicle
from pretraj import predict
from pretraj.merics import ADE, FDE
from pretraj import *

with open(REDUCED_NGSIM_JSON_PATH) as fp:
  l = json.load(fp)

observe_frames = 30
predict_frames = 18

ego = Vehicle(**l[1]['ego'])
pre = Vehicle(**l[1]['pre'])

ds_record, _, _ = predict.predict(ego, pre, observe_frames, predict_frames, 'adapt')
groundtruth_record = ego.space_headway_vector[observe_frames:observe_frames+predict_frames]
print(ADE(np.array(ds_record), np.array(groundtruth_record)))

ds_record, _, _ = predict.predict(ego, pre, observe_frames, predict_frames, 'IDM')
groundtruth_record = ego.space_headway_vector[observe_frames:observe_frames+predict_frames]
print(ADE(np.array(ds_record), np.array(groundtruth_record)))