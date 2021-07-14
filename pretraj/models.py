import numpy as np
import torch
from torch import nn
from typing import Callable, Any
from sklearn.linear_model import LinearRegression
from pretraj import vehicle
from pretraj import simulate
from pretraj import utils
import pretraj

def cache(fn):
  cache_info = {}
  def new_fn(
      ego: vehicle.Vehicle, 
      pre: vehicle.Vehicle, 
      observe_frames: int,
      **argvs: Any
  ) -> Callable:
    # key = (ego.vehicle_id, pre.vehicle_id, observe_frames, adapt)
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


@cache
def constant_velocity_model(**argvs) -> Callable:
  """Get constant velocity model control function."""
  def control_law(ego_state, pre_state):
    return 0

  return control_law


@cache
def IDM_model(pre: vehicle.Vehicle, **argvs) -> Callable:
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
def interaction_model(
    ego: vehicle.Vehicle,
    pre: vehicle.Vehicle,
    observe_frames: int,
    # adapt: bool = False,
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
def adaptation_interaction_model(
    ego: vehicle.Vehicle,
    pre: vehicle.Vehicle,
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

  lamb = 0.9
  # w = [kv, kg, c]
  w = np.load(pretraj.OFFLINE_REGRESSION)
  H = 1e-4 * np.eye(3) # 3 * 3
  update_times = 0
  ego_states = ego.states(1, observe_frames)

  def _adapt_control_law(ego_state, pre_state):
    nonlocal w, H, update_times
    dv = pre_state.v - ego_state.v
    gt = ego_state.ds - pre.vehicle_length # pre.vehicle_length is constant
    f = np.array([dv, gt, 1])[:, None] # 3 * 1
    a = w @ f # 1 * 1
    e = ego_states.pop(0).a - w @ f # 1 * 1
    # if np.abs(e[0]) < 0.5 and :
    newH = lamb * H + f @ f.T
    neww = w + e @ f.T @ np.linalg.inv(newH)
    # 1/2
    if np.abs(e[0]) < (1/2)**update_times and all(neww[0][:2]>0):
      update_times += 1
      H = newH; w = neww
    return a[0][0]

  simulate._simulate(ego.state(0), pre.states(1, observe_frames), _adapt_control_law)
  kv, kg, c = w[0]

  def control_law(ego_state, pre_state):
    dv = pre_state.v - ego_state.v
    gt = ego_state.ds - pre.vehicle_length # pre.vehicle_length is constant
    return kv * dv + kg * gt + c

  return control_law



@cache
def regularized_interaction_model(
    ego: vehicle.Vehicle,
    pre: vehicle.Vehicle,
    observe_frames: int,
    # adapt: bool = False,
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


  if np.linalg.matrix_rank(A) < A.shape[1]:
      params = np.array([0., 0, 0, g0])
  else:
      params = utils.solve_sdp(Ap, bp)
  kv, kg, g_star = params[0], params[1], params[3]

  def control_law(ego_state, pre_state):
    dv = pre_state.v - ego_state.v
    gt = ego_state.ds - pre.vehicle_length # pre.vehicle_length is constant
    return kv * dv + kg * (gt - g_star)

  return control_law

  

@cache
def neural_network(
    ego: vehicle.Vehicle,
    pre: vehicle.Vehicle,
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
  pre_a = pre.acc_vector[:observe_frames]
  Y = torch.FloatTensor(ego.acc_vector[:observe_frames][:,None]).to(device) # ground truth acceleration

  X = torch.FloatTensor(np.vstack([ds, ego_v, pre_v, pre_a]).T).to(device)

  model = torch.nn.Sequential(
    torch.nn.Linear(4, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1)
  ).to(device)

  lr = 1e-3
  n_epoch = 50
  opt = torch.optim.Adam(model.parameters(), lr=lr)

  # load from checkpoint
  checkpoint = torch.load(pretraj.PRETRAIN_MODEL_PATH)
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
    tensor = torch.FloatTensor([[ego_state.ds, ego_state.v, pre_state.v, pre_state.a]]).to(device)
    return model(tensor)[0][0].detach().cpu().numpy()

  return control_law



if __name__ == '__main__':
  utils.offline_regression(regularized=False)