import numpy as np
import torch
import cvxopt as cvx
import picos as pic
from pathlib import Path
from sklearn.linear_model import LinearRegression
import pretraj

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


def pretrain_neural_network():
  """pretrain the model on NGSIM and save to pretrain_model.pt"""
  print('='*10 + 'Pretraining Model' + '='*10)
  device = torch.device('cpu')

  # NUM_FRAMES = 200
  ds = np.hstack([ego.space_headway_vector[:pretraj.NUM_FRAMES] for ego, _ in pretraj.vehicle_pairs_list])
  ego_v = np.hstack([ego.vel_vector[:pretraj.NUM_FRAMES] for ego, _ in pretraj.vehicle_pairs_list])
  pre_v = np.hstack([pre.vel_vector[:pretraj.NUM_FRAMES] for _, pre in pretraj.vehicle_pairs_list])
  pre_a = np.hstack([pre.acc_vector[:pretraj.NUM_FRAMES] for _, pre in pretraj.vehicle_pairs_list])

  Y = torch.FloatTensor(
      np.vstack([ego.acc_vector[:pretraj.NUM_FRAMES][:,None] 
                 for ego, _ in pretraj.vehicle_pairs_list])).to(device)
  X = torch.FloatTensor(np.vstack([ds, ego_v, pre_v, pre_a]).T).to(device)
  permutation = torch.randperm(Y.size()[0])
  Y = Y[permutation]; X = X[permutation]

  split_point = X.size()[0] // 4
  Y_train = Y[:split_point]; X_train = X[:split_point]
  Y_test = Y[split_point:];  X_test = X[split_point:]
  

  model = torch.nn.Sequential(
    torch.nn.Linear(4, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1)
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
    print('evaluation:', eval)
    if eval < lowest: cnt = 0; lowest = eval
    if cnt > 10: break

  Path(pretraj.CHECKPOINT_DIR).mkdir(exist_ok=True)
  torch.save({'model_state_dict': model.state_dict(),
              'optimizer_state_dict': opt.state_dict(),
             }, pretraj.PRETRAIN_MODEL_PATH)



def offline_regression(regularized=False):
  """offline regression of the whole NGSIM on interaction model"""
  dv = np.hstack([(pre.vel_vector - ego.vel_vector)[:pretraj.NUM_FRAMES] 
      for ego, pre in pretraj.vehicle_pairs_list])
  gt = np.hstack([(ego.space_headway_vector - pre.vehicle_length)[:pretraj.NUM_FRAMES] 
      for ego, pre in pretraj.vehicle_pairs_list])
  Y = np.hstack([ego.acc_vector[:pretraj.NUM_FRAMES] 
      for ego, _ in pretraj.vehicle_pairs_list])
#   dv = np.hstack([(pre.vel_vector - ego.vel_vector)[:pretraj.NUM_FRAMES] 
#       for ego, pre in pretraj.vehicle_pairs_list][:100])
#   gt = np.hstack([(ego.space_headway_vector - pre.vehicle_length)[:pretraj.NUM_FRAMES] 
#       for ego, pre in pretraj.vehicle_pairs_list][:100])
#   Y = np.hstack([ego.acc_vector[:pretraj.NUM_FRAMES] 
#       for ego, _ in pretraj.vehicle_pairs_list][:100])

  if not regularized:
    X = np.vstack([dv, gt]).T.reshape(-1,2)
    reg = LinearRegression(positive=True).fit(X, Y)
    kv, kg = reg.coef_
    c = reg.intercept_

  else:
    alpha = 1.
    beta = 1.
    dv = dv[:100]; gt = gt[:100]; Y = Y[:100]
    g0 = gt.mean()
    A = np.array([dv, gt, 0*gt-1]).T
    Ap = np.vstack((np.hstack((A, 0*A[:, [0]])), 
                    np.array([[0, 0, 0, np.sqrt(alpha)],
                              [np.sqrt(beta)*g0, 0, 0, 0],
                              [0, np.sqrt(beta)*g0, 0, 0]])))
    bp = np.hstack((Y, np.sqrt(alpha)*g0, 0, 0))

    if np.linalg.matrix_rank(A) < A.shape[1]:
        params = np.array([0., 0, 0, g0])
    else:
        params = solve_sdp(Ap, bp)
    kv, kg, gs = params[0], params[1], params[3]
    c = -kg*gs

  w = np.array([[kv, kg, c]]) # 1 * 3
  print(w)
  Path(pretraj.CHECKPOINT_DIR).mkdir(exist_ok=True)
  np.save(pretraj.OFFLINE_REGRESSION, w)