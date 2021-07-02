import numpy as np
from vehicle import Vehicle

def IDM_simulation(
    ego_vehicle: Vehicle, 
    pre_vehicle: Vehicle, 
    observe_frames: int, 
    predict_frames: int,
) -> np.ndarray:
  pass

def adapt_simulation(
    ego_vehicle: Vehicle, 
    pre_vehicle: Vehicle, 
    observe_frames: int, 
    predict_frames: int,
    model='adapt'
) -> np.ndarray:
  pass

def predict(
    ego_vehicle: Vehicle, 
    pre_vehicle: Vehicle, 
    observe_frames: int, 
    predict_frames: int,
    model='adapt'
) -> np.ndarray:
  assert model in ('IDM', 'adapt'), "model should be IDM or adapt"
  return IDM_simulation(ego_vehicle, pre_vehicle) if model is 'IDM' \
  else adapt_simulation(ego_vehicle, pre_vehicle)