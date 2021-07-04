import json
import numpy as np
from copy import copy
from typing import Tuple, List, Callable
from numpy.testing._private.utils import print_assert_equal
from sklearn.linear_model import LinearRegression
from pretraj.vehicle import Vehicle, State

from pretraj import *


def simulate(
    ego_state: State, 
    pre_states: List[State],
    control_law: Callable[[State, State], float]
) -> Tuple[List[int], List[int], List[int], bool]:
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
    # control_law is different for different models
    ego_state.a = control_law(ego_state, pre_state)
    if ego_state.v <= 0:
      hard_braking = True
      state_record.extend([State(ds=ego_state.ds)] * (len(pre_states)-i))
      break
    state_record.append(copy(ego_state))

  return tuple(zip(*[(state.ds, state.v, state.a) for state in state_record])) +  tuple([hard_braking])


def IDM_simulation(
    ego: Vehicle, 
    pre: Vehicle, 
    observe_frames: int, 
    predict_frames: int,
) -> Tuple[List[int], List[int], List[int], bool]:
  """Simulate the trajectory of ego vehicle based on IDM model (fixed).

  Args:
    ego: instance of ego vehicle
    pre: instance of precede vehicle
    observe_frames: the number of observed frames before prediction
    predict_frames: the number of predicted frames after observation.

  Returns:
    A tuple of lists of predicted ego vehicle states (headway, velocity, acceleration)
    and indication whether the car hard brakes.
  """
  Metre_Foot = 3.2808399
  KilometrePerHour_FootPerSecond = 1000 * Metre_Foot / 3600

  def control_law(ego_state, pre_state):
    # coefficients for 'car' from KESTING's page 4588
    v0 = 120 * KilometrePerHour_FootPerSecond
    delta = 4
    T = 1.5
    s0 = 2 * Metre_Foot
    a = 1.4 * Metre_Foot
    b = 2 * Metre_Foot
    c = 0.99

    dv = ego_state.v - pre_state.v
    ss = s0 + ego_state.v * T + (ego_state.v * dv) / (2*(a*b)**.5)
    s = ego_state.ds - pre.vehicle_length
    return a * (1 - (ego_state.v/v0)**delta - (ss/s)**2)

  return simulate(
      ego.state(observe_frames-1),
      pre.states(observe_frames, predict_frames),
      control_law)
  

def adapt_simulation(
    ego: Vehicle,
    pre: Vehicle,
    observe_frames: int,
    predict_frames: int,
) -> Tuple[List[int], List[int], List[int], bool]:
  """Simulate the trajectory of ego vehicle based on adaptation model.

  Args:
    ego: instance of ego vehicle
    pre: instance of precede vehicle
    observe_frames: the number of observed frames before prediction
    predict_frames: the number of predicted frames after observation.

  Returns:
    A tuple of lists of predicted ego vehicle states (headway, velocity, acceleration)
    and indication whether the car hard brakes.
  """
  # Linear regressor to learn coefficients kv, kg, gs
  dv = (pre.vel_vector - ego.vel_vector)[:observe_frames]
  gt = (ego.space_headway_vector - pre.vehicle_length)[:observe_frames]
  Y = ego.acc_vector[:observe_frames] # ground truth acceleration

  X = np.vstack([dv, gt]).T.reshape(-1,2)
  reg = LinearRegression(positive=True).fit(X, Y)
  kv, kg = reg.coef_
  gs = -reg.intercept_/(kg if kg else 1)

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
) -> Tuple[List[int], List[int], List[int], bool]:
  # TODO
  pass


def predict(
    ego: Vehicle, 
    pre: Vehicle,
    observe_frames: int,
    predict_frames: int,
    model='adapt'
) -> Tuple[List[int], List[int], List[int], bool]:
  """predict """
  assert model in ('IDM', 'adapt'), 'model should be IDM or adapt'
  assert observe_frames > 0 and predict_frames > 0 and observe_frames + predict_frames <= NUM_FRAMES,\
      f'observe_frames > 0 and observe_frames < 0 and observe_frames + predict_frames <= {NUM_FRAMES}'
  return IDM_simulation(ego, pre, observe_frames, predict_frames) if model is 'IDM' \
  else adapt_simulation(ego, pre, observe_frames, predict_frames)

