import numpy as np
import json
from copy import copy
from typing import Tuple, Callable, List
from pretraj import vehicle
from pretraj import models
from pretraj import metrics
import pretraj

models_list = [
    'constant velocity', 
    'IDM', 
    'interaction',
    'interaction (adaptation)',
    'interaction (regularization)',
    # 'interaction (adaptation + regularization)',
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
    model='constant velocity'
) -> Tuple[List[int], List[int], List[int], bool]:
  """simulate"""
  models_dict = {
      'constant velocity': models.constant_velocity_model, 
      'IDM': models.IDM_model, 
      'interaction': models.interaction_model,
      'interaction (adaptation)': models.interaction_model,
      'interaction (regularization)': models.regularized_interaction_model,
      'interaction (adaptation + regularization)': models.regularized_interaction_model,
      'neural network': models.neural_network}
  assert model in models_dict.keys(), f'model should be {models_dict.keys()}'
  assert observe_frames > 0 and predict_frames > 0 and \
      observe_frames + predict_frames <= pretraj.NUM_FRAMES, \
      f'observe_frames > 0 and observe_frames < 0 and observe_frames + predict_frames <= {pretraj.NUM_FRAMES}'

  control_law = models_dict[model](
      ego=ego, 
      pre=pre, 
      observe_frames=observe_frames,
      adapt=True if 'adapt' in model else False)

  return _simulate(
      ego.state(observe_frames-1),
      pre.states(observe_frames, predict_frames),
      control_law)



# test
if __name__ == '__main__':
  with open(pretraj.REDUCED_NGSIM_JSON_PATH) as fp:
    pair_info = json.load(fp)

  ego = vehicle.Vehicle(**pair_info[1]['ego'])
  pre = vehicle.Vehicle(**pair_info[1]['pre'])

  from pretraj.metrics import ADE, FDE

  observe_frames = 100

  predict_frames = 50

  groundtruth_record = ego.space_headway_vector[observe_frames:observe_frames+predict_frames]

  models_list = ['interaction', 'interaction (adaptation)', 'interaction (regularization)', 'interaction (adaptation + regularization)']
  for model in models_list:
    hard_braking, (ds_record, _, _) = simulate(ego, pre, observe_frames, predict_frames, model)
    result = ADE(np.array(ds_record), np.array(groundtruth_record))
    print(model, result)