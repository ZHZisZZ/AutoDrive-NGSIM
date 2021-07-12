import numpy as np
import json
from copy import copy
from typing import Tuple, Callable, List
from pretraj import vehicle
from pretraj import models
import pretraj

models_list = [
    'constant velocity', 
    'IDM', 
    'adaptation', 
    'regularized adaptation',
    'neural network'
]

def _simulate(
    ego_state: vehicle.State, 
    pre_states: List[vehicle.State],
    control_law: Callable[[vehicle.State, vehicle.State], float]
) -> Tuple[bool, Tuple[List[int], List[int], List[int]]]:
  """Simulate the trajectory of ego vehicle according to states and control law.

  Args:
    ego_state: the state of ego vehicle at the last observed frame.
    pre_states: the states of precede vehicles over all predicted frames.
    control_law: a function that maps current states of two vehicles to the
      acceleration of the ego vehicle.

  Returns:
    A tuple of lists of predicted ego vehicle states (headway, velocity, acceleration)
    and indication whether the car hard brakes.
  """
  state_record = []
  dt = .1
  hard_braking = False
  for i, pre_state in enumerate(pre_states):
    dv = pre_state.v - ego_state.v
    da = pre_state.a - ego_state.a

    ego_state.ds += dv * dt + .5 * da * dt**2
    ego_state.v += (ego_state.a) * dt
    if ego_state.v <= 0:
      hard_braking = True
      ego_state.v = 0
      # does not change ego_state.ds for simplicity
    ego_state.a = control_law(ego_state, pre_state)

    state_record.append(copy(ego_state))

  return hard_braking, tuple(zip(*[(state.ds, state.v, state.a) for state in state_record]))


def simulate(
    ego: vehicle.Vehicle, 
    pre: vehicle.Vehicle,
    observe_frames: int,
    predict_frames: int,
    model='adapt'
) -> Tuple[List[int], List[int], List[int], bool]:
  """simulate"""
  models_dict = {
      'constant velocity': models.constant_velocity_model, 
      'IDM': models.IDM_model, 
      'adaptation': models.adaptation_model, 
      'regularized adaptation': models.regularized_adaptation_model,
      'neural network': models.neural_network}
  assert model in models_dict.keys(), f'model should be {models_dict.keys()}'
  assert observe_frames > 0 and predict_frames > 0 and \
      observe_frames + predict_frames <= pretraj.NUM_FRAMES, \
      f'observe_frames > 0 and observe_frames < 0 and observe_frames + predict_frames <= {pretraj.NUM_FRAMES}'

  control_law = models_dict[model](
      ego=ego, 
      pre=pre, 
      observe_frames=observe_frames)

  return _simulate(
      ego.state(observe_frames-1),
      pre.states(observe_frames, predict_frames),
      control_law)



if __name__ == '__main__':
  with open(pretraj.REDUCED_NGSIM_JSON_PATH) as fp:
    pair_info = json.load(fp)

  ego = vehicle.Vehicle(**pair_info[10]['ego'])
  pre = vehicle.Vehicle(**pair_info[10]['pre'])

  from pretraj.metrics import ADE, FDE

  observe_frames = 100

  predict_frames = 50

  groundtruth_record = ego.space_headway_vector[observe_frames:observe_frames+predict_frames]


  hard_braking, (ds_record, _, _) = simulate(ego, pre, observe_frames, predict_frames, 'neural network')
  result = ADE(np.array(ds_record), np.array(groundtruth_record))
  print('neural network:', result)
  hard_braking, (ds_record, _, _) = simulate(ego, pre, observe_frames, predict_frames, 'adaptation')
  result = ADE(np.array(ds_record), np.array(groundtruth_record))
  print('adapt:', result)
  hard_braking, (ds_record, _, _) = simulate(ego, pre, observe_frames, predict_frames, 'regularized adaptation')
  result = ADE(np.array(ds_record), np.array(groundtruth_record))
  print('regularized adapt:', result)
  hard_braking, (ds_record, _, _) = simulate(ego, pre, observe_frames, predict_frames, 'constant velocity')
  result = ADE(np.array(ds_record), np.array(groundtruth_record))
  print('constantv:', result)