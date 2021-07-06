import json
import numpy as np
import torch
from torch import nn
from copy import copy
from typing import Tuple, List, Callable
from sklearn.linear_model import LinearRegression
from pretraj.vehicle import Vehicle, State

from pretraj import *


def simulate(
    ego_state: State, 
    pre_states: List[State],
    control_law: Callable[[State, State], float]
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


def constantv_model(**argvs) -> Callable:
  """Get constant velocity model control function."""
  def control_law(ego_state, pre_state):
    return 0

  return control_law


def IDM_model(pre: Vehicle, **argvs) -> Callable:
  """Get IDM model control function."""
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

  return control_law
  

def adapt_model(
    ego: Vehicle,
    pre: Vehicle,
    observe_frames: int,
    **argvs,
) -> Callable:
  """Get adapt model control function.

  Args:
    ego: instance of ego vehicle
    pre: instance of precede vehicle
    observe_frames: the number of observed frames before prediction

  Returns:
    adapt model control function
  """
  # Linear regressor to learn coefficients kv, kg, gs
  dv = (pre.vel_vector - ego.vel_vector)[:observe_frames]
  gt = (ego.space_headway_vector - pre.vehicle_length)[:observe_frames]
  Y = ego.acc_vector[:observe_frames] # ground truth acceleration

  X = np.vstack([dv, gt]).T.reshape(-1,2)
  reg = LinearRegression(positive=True).fit(X, Y)
  kv, kg = reg.coef_
  c = reg.intercept_
  # gs = -reg.intercept_/(kg if kg else 1)

  def control_law(ego_state, pre_state):
    dv = pre_state.v - ego_state.v
    gt = ego_state.ds - pre.vehicle_length # pre.vehicle_length is constant
    return kv * dv + kg * gt + c

  return control_law


def nn_model(
    ego: Vehicle,
    pre: Vehicle,
    observe_frames: int,
    **argvs,
) -> Callable:
  """Get neural network model control function

  Args:
    ego: instance of ego vehicle
    pre: instance of precede vehicle
    observe_frames: the number of observed frames before prediction

  Returns:
    neural network model control function
  """
  # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  device = torch.device('cpu')

  ds = ego.space_headway_vector[:observe_frames]
  ego_v = ego.vel_vector[:observe_frames]
  pre_v = pre.vel_vector[:observe_frames]
  Y = torch.FloatTensor(ego.acc_vector[:observe_frames][:,None]).to(device) # ground truth acceleration

  X = torch.FloatTensor(np.vstack([ds, ego_v, pre_v]).T).to(device)

  model = torch.nn.Sequential(
    torch.nn.Linear(3, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1)
  ).to(device)

  lr = 1e-3
  lowest = float("inf")
  cnt = 0
  opt = torch.optim.Adam(model.parameters(), lr=lr)
  for _ in range(1000):
    pred = model(X)
    # loss = torch.nn.MSELoss()(Y, pred)
    loss = nn.L1Loss()(Y, pred)

    cnt += 1
    if loss < lowest: cnt = 0; lowest = loss
    if cnt > 50: break

    # print(loss)
    loss.backward()
    opt.step()
    opt.zero_grad()

  def control_law(ego_state, pre_state):
    tensor = torch.FloatTensor([[ego_state.ds, ego_state.v, pre_state.v]]).to(device)
    return model(tensor)[0][0].detach().cpu().numpy()

  # TODO: test
  # tensor = torch.FloatTensor([[78.97, 25.42, 34.91]]).to(device)
  # print(model(tensor)[0][0].detach().cpu().numpy())
  # print(np.hstack([X, Y.numpy()]))

  return control_law


def probability_model(
    ego: Vehicle,
    pre: Vehicle,
    observe_frames: int, 
) -> Callable:
  pass


def predict(
    ego: Vehicle, 
    pre: Vehicle,
    observe_frames: int,
    predict_frames: int,
    model='adapt'
) -> Tuple[List[int], List[int], List[int], bool]:
  """predict """
  model_dict = {'constantv': constantv_model, 'IDM': IDM_model, 'adapt': adapt_model, 'nn':nn_model}
  assert model in model_dict.keys(), f'model should be {model_dict.keys()}'
  assert observe_frames > 0 and predict_frames > 0 and \
      observe_frames + predict_frames <= NUM_FRAMES, \
      f'observe_frames > 0 and observe_frames < 0 and observe_frames + predict_frames <= {NUM_FRAMES}'

  control_law = model_dict[model](
      ego=ego, 
      pre=pre, 
      observe_frames=observe_frames)

  return simulate(
      ego.state(observe_frames-1),
      pre.states(observe_frames, predict_frames),
      control_law)



if __name__ == '__main__':
  with open(REDUCED_NGSIM_JSON_PATH) as fp:
    pair_info = json.load(fp)

  ego = Vehicle(**pair_info[7]['ego'])
  pre = Vehicle(**pair_info[7]['pre'])

  from pretraj.merics import ADE, FDE

  observe_frames = 100
  predict_frames = 50

  groundtruth_record = ego.space_headway_vector[observe_frames:observe_frames+predict_frames]

  hard_braking, (ds_record, _, _) = predict(ego, pre, observe_frames, predict_frames, 'nn')
  result = ADE(np.array(ds_record), np.array(groundtruth_record))
  print('nn:', result)
  hard_braking, (ds_record, _, _) = predict(ego, pre, observe_frames, predict_frames, 'adapt')
  result = ADE(np.array(ds_record), np.array(groundtruth_record))
  print('adapt:', result)
  hard_braking, (ds_record, _, _) = predict(ego, pre, observe_frames, predict_frames, 'constantv')
  result = ADE(np.array(ds_record), np.array(groundtruth_record))
  print('constantv:', result)