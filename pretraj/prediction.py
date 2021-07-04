import json
import numpy as np
from typing import List, Callable
from sklearn.linear_model import LinearRegression
from pretraj.vehicle import Vehicle, State

from pretraj import *


def simulate(
    ego_state: State, 
    pre_states: List[State],
    control_law: Callable[[State, State], float]
) -> List[int]:
  """Simulate the trajectory of ego vehicle according to states and control law.

  Args:
    ego_state: the state of ego vehicle at the last observed frame.
    pre_states: the states of precede vehicles over all predicted frames.
    control_law: a function that maps current states of two vehicles to the
      acceleration of the ego vehicle.

  Returns:
    A list of predicted space headway of the ego car, which serve as 
      trajectory info.
  """
  ds_record = []
  dt = .1
  for pre_state in pre_states:
    dv = pre_state.v - ego_state.v
    da = pre_state.a - ego_state.a

    ego_state.ds += dv * dt + .5 * da * dt**2
    ego_state.v += (ego_state.a) * dt

    # control_law is different for different models
    ego_state.a = control_law(ego_state, pre_state)
    ds_record.append(ego_state.ds)

  return ds_record


def IDM_simulation(
    ego: Vehicle, 
    pre: Vehicle, 
    observe_frames: int, 
    predict_frames: int,
) -> np.ndarray:
  pass


def adapt_simulation(
    ego: Vehicle,
    pre: Vehicle,
    observe_frames: int,
    predict_frames: int,
) -> np.ndarray:
  """Simulate the trajectory of ego vehicle based on adaptation model.
    
  """
  # Linear regressor to learn coefficients kv, kg, gs
  dv = (pre.vel_vector - ego.vel_vector)[:observe_frames]
  gt = (ego.space_headway_vector - pre.vehicle_length)[:observe_frames]
  Y = ego.acc_vector[:observe_frames] # ground truth acceleration

  X = np.vstack([dv, gt]).T.reshape(-1,2)
  reg = LinearRegression(positive=True).fit(X, Y)
  kv, kg = reg.coef_; gs = -reg.intercept_/kg

  print(kv, kg, gs)

  def control_law(ego_state, pre_state):
    dv = pre_state.v - ego_state.v
    gt = ego_state.ds - pre.vehicle_length # pre.vehicle_length is constant
    return kv * dv + kg * (gt - gs)
  
  return simulate(
      ego.state(observe_frames-1),
      pre.states(observe_frames, predict_frames),
      control_law)


def model_free_simulation(
    ego: Vehicle,
    pre: Vehicle,
    observe_frames: int, 
    predict_frames: int,
) -> np.ndarray:
  pass


def predict(
    ego: Vehicle, 
    pre: Vehicle,
    observe_frames: int,
    predict_frames: int,
    model='adapt'
) -> np.ndarray:
  assert model in ('IDM', 'adapt'), "model should be IDM or adapt"
  return IDM_simulation(ego, pre, observe_frames, predict_frames) if model is 'IDM' \
  else adapt_simulation(ego, pre, observe_frames, predict_frames)


if __name__ == '__main__': 
  with open(REDUCED_NGSIM_JSON_PATH) as fp:
    l = json.load(fp)

  # 3, fail on 1e-4
  ego = Vehicle(**l[11]['ego'])
  pre = Vehicle(**l[11]['pre'])

  # print(ego.vel_vector)
  # print(pre.vel_vector)

  observe_frames = 30
  predict_frames = 18
  ds_record = adapt_simulation(ego, pre, observe_frames, predict_frames)
  print(ds_record)
  print(ego.space_headway_vector[observe_frames:observe_frames+predict_frames])