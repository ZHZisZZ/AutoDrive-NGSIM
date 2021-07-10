import json
import numpy as np
import torch
import cvxopt as cvx
import picos as pic
from torch import nn
from pathlib import Path
from copy import copy
from typing import Tuple, Callable, List, Any
from sklearn.linear_model import LinearRegression
from pretraj.vehicle import Vehicle, State

from pretraj import *


def cache(fn):
  cache_info = {}
  def new_fn(
      ego: Vehicle, 
      pre: Vehicle, 
      observe_frames: int, 
      **argvs: Any
  ) -> Callable:
    # print(cache_info)
    key = (ego.vehicle_id, pre.vehicle_id, observe_frames)
    if key in cache_info:
      return cache_info[key]
    result = fn(ego=ego,
                pre=pre, 
                observe_frames=observe_frames, 
                **argvs)
    cache_info[key] = result
    return result
  return new_fn


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


@cache
def constant_velocity_model(**argvs) -> Callable:
  """Get constant velocity model control function."""
  def control_law(ego_state, pre_state):
    return 0

  return control_law


@cache
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
  

@cache
def adaptation_model(
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


@cache
def regularized_adaptation_model(
    ego: Vehicle,
    pre: Vehicle,
    observe_frames: int,
    **argvs,
) -> Callable:
  """Get regularized adapt model control function.

  Args:
    ego: instance of ego vehicle
    pre: instance of precede vehicle
    observe_frames: the number of observed frames before prediction

  Returns:
    adapt model control function
  """
  alpha = 1.
  beta = 1.
  dv = (pre.vel_vector - ego.vel_vector)[:observe_frames]
  gt = (ego.space_headway_vector - pre.vehicle_length)[:observe_frames]
  g0 = gt.mean()
  A = np.array([dv, gt, 0*gt-1]).T
  Ap = np.vstack((np.hstack((A, 0*A[:, [0]])), 
                  np.array([[0, 0, 0, np.sqrt(alpha)],
                            [np.sqrt(beta)*g0, 0, 0, 0],
                            [0, np.sqrt(beta)*g0, 0, 0]])))
  bp = np.hstack((ego.acc_vector[:observe_frames], np.sqrt(alpha)*g0, 0, 0))


  def solve_sdp(A, b):
    """Solve dp problem."""
    # x = [kv, kg, u, g]
    # kg*g - u = 0
    K_UB = 20
    EPS = 1e-5
    m = A.shape[1]
    B = np.zeros((m, m))
    B[1, 3] = B[3, 1] = 1
    q = np.array([0, 0, -1, 0])
    A, b, B, q = [cvx.matrix(arr.copy()) for arr in [A, b, B, q]]

    prob = pic.Problem()
    x = prob.add_variable('x', m)
    X = prob.add_variable('X', (m, m), vtype='symmetric')
    prob.add_constraint(x >= EPS)
    prob.add_constraint(((1 & x.T) // (x & X)) >> 0)
    prob.add_constraint(0.5*(X | B) + q.T*x  == 0)

    prob.set_objective('min', (X | A.T * A) - 2 * b.T * A * x)

    try:
        prob.solve(verbose=0, solver='cvxopt', solve_via_dual=False, tol=EPS/2)
    except ValueError:
        print(A)
        print('retrying')
        prob.solve(verbose=0, solver='cvxopt', solve_via_dual=False, tol=0.1)
    x_hat = np.array(x.value).T[0]
    assert (x_hat >= EPS-1e-2).all() and (x_hat[:2] <= K_UB+1e-2).all(), '{}'.format(x)
    return x_hat


  if np.linalg.matrix_rank(A) < A.shape[1]:
      params = np.array([0., 0, 0, g0])
  else:
      params = solve_sdp(Ap, bp)
  kv, kg, g_star = params[0], params[1], params[3]

  def control_law(ego_state, pre_state):
    dv = pre_state.v - ego_state.v
    gt = ego_state.ds - pre.vehicle_length # pre.vehicle_length is constant
    return kv * dv + kg * (gt - g_star)

  return control_law
  

@cache
def neural_network(
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
  n_epoch = 50
  opt = torch.optim.Adam(model.parameters(), lr=lr)

  # load from checkpoint
  checkpoint = torch.load(PRETRAIN_MODEL_PATH)
  model.load_state_dict(checkpoint['model_state_dict'])
  opt.load_state_dict(checkpoint['optimizer_state_dict'])
  
  for _ in range(n_epoch):
    pred = model(X)
    # loss = torch.nn.MSELoss()(Y, pred)
    loss = nn.L1Loss()(Y, pred)

    # print(loss)
    loss.backward()
    opt.step()
    opt.zero_grad()

  def control_law(ego_state, pre_state):
    tensor = torch.FloatTensor([[ego_state.ds, ego_state.v, pre_state.v]]).to(device)
    return model(tensor)[0][0].detach().cpu().numpy()

  return control_law



def pretrain_neural_network():
  # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  device = torch.device('cpu')

  with open(REDUCED_NGSIM_JSON_PATH) as fp:
      pair_info = json.load(fp)

  vehicle_pairs_list = \
    [(Vehicle(**pair_info[i]['ego']),
      Vehicle(**pair_info[i]['pre']))
      for i in range(len(pair_info) // 4)]

  # NUM_FRAMES = 200
  ds = np.hstack([ego.space_headway_vector[:NUM_FRAMES] for ego, _ in vehicle_pairs_list])
  ego_v = np.hstack([ego.vel_vector[:NUM_FRAMES] for ego, _ in vehicle_pairs_list])
  pre_v = np.hstack([pre.vel_vector[:NUM_FRAMES] for _, pre in vehicle_pairs_list])

  Y = torch.FloatTensor(
      np.vstack([ego.acc_vector[:NUM_FRAMES][:,None] 
                 for ego, _ in vehicle_pairs_list])).to(device)
  X = torch.FloatTensor(np.vstack([ds, ego_v, pre_v]).T).to(device)
  permutation = torch.randperm(Y.size()[0])
  Y = Y[permutation]; X = X[permutation]

  split_point = X.size()[0] // 4
  Y_train = Y[:split_point]; X_train = X[:split_point]
  Y_test = Y[split_point:];  X_test = X[split_point:]
  

  model = torch.nn.Sequential(
    torch.nn.Linear(3, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1)
  ).to(device)


  lr = 1e-3
  n_epochs = 100
  batch_size = 512 * 2
  opt = torch.optim.Adam(model.parameters(), lr=lr)
  # obj = torch.nn.MSELoss()
  obj = torch.nn.L1Loss()

  lowest = float("inf")
  cnt = 0
  for epoch in range(n_epochs):
    permutation = torch.randperm(X_train.size()[0])

    for i in range(0, X_train.size()[0], batch_size):
      indices = permutation[i:i+batch_size]
      batch_x, batch_y = X_train[indices], Y_train[indices]
      pred = model(batch_x)
      # loss = torch.nn.MSELoss()(batch_y, pred)
      loss = obj(batch_y, pred)

      loss.backward()
      opt.step()
      opt.zero_grad()

    cnt += 1
    eval = obj(Y_test, model(X_test))
    if eval < lowest: cnt = 0; lowest = eval
    if cnt > 10: break

  Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
  torch.save({'model_state_dict': model.state_dict(),
              'optimizer_state_dict': opt.state_dict(),
             }, PRETRAIN_MODEL_PATH)


def predict(
    ego: Vehicle, 
    pre: Vehicle,
    observe_frames: int,
    predict_frames: int,
    model='adapt'
) -> Tuple[List[int], List[int], List[int], bool]:
  """predict """
  model_dict = {
      'constant velocity': constant_velocity_model, 
      'IDM': IDM_model, 
      'adaptation': adaptation_model, 
      'regularized adaptation': regularized_adaptation_model,
      'neural network': neural_network}
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

  ego = Vehicle(**pair_info[10]['ego'])
  pre = Vehicle(**pair_info[10]['pre'])

  from pretraj.merics import ADE, FDE

  observe_frames = 100

  predict_frames = 50

  groundtruth_record = ego.space_headway_vector[observe_frames:observe_frames+predict_frames]


  hard_braking, (ds_record, _, _) = predict(ego, pre, observe_frames, predict_frames, 'neural network')
  result = ADE(np.array(ds_record), np.array(groundtruth_record))
  print('neural network:', result)
  # hard_braking, (ds_record, _, _) = predict(ego, pre, observe_frames, predict_frames, 'adapt')
  # result = ADE(np.array(ds_record), np.array(groundtruth_record))
  # print('adapt:', result)
  # hard_braking, (ds_record, _, _) = predict(ego, pre, observe_frames, predict_frames, 'regularized_adapt')
  # result = ADE(np.array(ds_record), np.array(groundtruth_record))
  # print('regularized adapt:', result)
  # hard_braking, (ds_record, _, _) = predict(ego, pre, observe_frames, predict_frames, 'constantv')
  # result = ADE(np.array(ds_record), np.array(groundtruth_record))
  # print('constantv:', result)


"""
if __name__ == '__main__':
  pretrain_neural_network()
"""