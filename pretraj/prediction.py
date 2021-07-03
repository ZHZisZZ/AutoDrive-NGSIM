import json
import torch
import numpy as np
from pretraj.vehicle import Vehicle

from pretraj import *

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
  # train
  dv = torch.tensor((pre.vel_vector - ego.vel_vector)[:observe_frames])
  gt = torch.tensor((ego.space_headway_vector - pre.vehicle_length)[:observe_frames])
  a_tru = torch.tensor(ego.acc_vector[:observe_frames])

  kv = torch.tensor(0.0, requires_grad=True)
  kg = torch.tensor(0.0, requires_grad=True)
  gs = torch.tensor(0.0, requires_grad=True)

  lr = 1e-4
  epoch = int(1e4)
  opt = torch.optim.SGD([kv,kg,gs], lr=lr)

  for _ in range(epoch):
    # a_pre = kv*dv + kg*(gt - gs)
    a_pre = kv*dv + kg*(gt - gs)
    loss = torch.sum((a_tru - a_pre)**2)
    # print(a_tru)
    print(loss)
    loss.backward()
    opt.step()
    opt.zero_grad()

  print(kv, kg, gs)
  print(a_tru)
  print(kv*dv + kg*(gt - gs))


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
  ego = Vehicle(**l[10]['ego'])
  pre = Vehicle(**l[10]['pre'])

  # print(ego.vel_vector)
  # print(pre.vel_vector)

  adapt_simulation(ego, pre, 8, 8)