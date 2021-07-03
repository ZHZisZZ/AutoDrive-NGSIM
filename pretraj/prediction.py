import json
from os import name
import numpy as np
from collections import namedtuple
from pretraj.vehicle import Vehicle

from pretraj import *

def IDM_simulation(
    ego: Vehicle, 
    pre: Vehicle, 
    observe_frames: int, 
    predict_frames: int,
) -> np.ndarray:
  pass

# def adapt_simulation(
#     ego: Vehicle, 
#     pre: Vehicle, 
#     observe_frames: int, 
#     predict_frames: int,
# ) -> np.ndarray:
#   # linear regressor
#   dv = torch.tensor((pre.vel_vector - ego.vel_vector)[:observe_frames])
#   gt = torch.tensor((ego.space_headway_vector - pre.vehicle_length)[:observe_frames])
#   a_tru = torch.tensor(ego.acc_vector[:observe_frames])

#   kv = torch.tensor(0.0, requires_grad=True)
#   kg = torch.tensor(0.0, requires_grad=True)
#   gs = torch.tensor(0.0, requires_grad=True)

#   lr = 1e-4
#   epoch = int(1e4)
#   opt = torch.optim.SGD([kv,kg,gs], lr=lr)

#   for _ in range(epoch):
#     # pre_a = kv*dv + kg*(gt - gs)
#     pre_a = kv*dv + kg*(gt - gs)
#     loss = torch.sum((a_tru - pre_a)**2)
#     # print(a_tru)
#     print(loss)
#     loss.backward()
#     opt.step()
#     opt.zero_grad()

#   print(kv, kg, gs)
#   print(a_tru)
#   print(kv*dv + kg*(gt - gs))

def adapt_simulation(
    ego: Vehicle, 
    pre: Vehicle, 
    observe_frames: int, 
    predict_frames: int,
) -> np.ndarray:
  # linear regressor
  dv = (pre.vel_vector - ego.vel_vector)[:observe_frames]
  gt = (ego.space_headway_vector - pre.vehicle_length)[:observe_frames]
  Y = ego.acc_vector[:observe_frames] # acceleration

  X = np.vstack([dv, gt, np.ones(observe_frames)])
  c = Y @ X.T @ np.linalg.inv(X @ X.T)
  kv = c[0]; kg = c[1]; gs = -c[2]/kg

  print(kv, kg, gs)

  # simulation
  class State(object):
    def __init__(self, a, v, ds):
      self.a = a; self.v = v; self.ds = ds

  pre_states = [State(
      pre.acc_vector[i], 
      pre.vel_vector[i], 
      None) 
      for i in range(observe_frames, observe_frames+predict_frames)]

  ego_state = State(
      ego.acc_vector[observe_frames-1], 
      ego.vel_vector[observe_frames-1], 
      ego.space_headway_vector[observe_frames-1])

  ds_record = []
  dt = .1
  for pre_state in pre_states:
    dv = pre_state.v - ego_state.v
    da = pre_state.a - ego_state.a

    ego_state.ds += dv * dt + .5 * da * dt**2
    ego_state.v += (ego_state.a) * dt
    # print(ego_state.a)
    # print(ego_state.v)
    # print(pre_state.v)

    gt = ego_state.ds - pre.vehicle_length
    ego_state.a = kv * dv + kg * (gt - gs) # different for different models
    ds_record.append(ego_state.ds)

  print(ds_record)
  print(ego.space_headway_vector[observe_frames:observe_frames+predict_frames])


def predict(
    ego: Vehicle, 
    pre: Vehicle, 
    observe_frames: int, 
    predict_frames: int,
    model='adapt'
) -> np.ndarray:
  assert model in ('IDM', 'adapt'), "model should be IDM or adapt"
  return IDM_simulation(ego, pre) if model is 'IDM' \
  else adapt_simulation(ego, pre)


if __name__ == '__main__': 
  with open(REDUCED_NGSIM_JSON_PATH) as fp:
    l = json.load(fp)

  # 3, fail on 1e-4
  ego = Vehicle(**l[127]['ego'])
  pre = Vehicle(**l[127]['pre'])

  # print(ego.vel_vector)
  # print(pre.vel_vector)

  adapt_simulation(ego, pre, 45, 3)