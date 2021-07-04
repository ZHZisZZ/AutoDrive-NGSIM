import numpy as np


def ADE(
    traj_1: np.ndarray, 
    traj_2: np.ndarray
):
  return np.sum(np.abs((traj_1 - traj_2)))


def FDE(
    traj_1: np.ndarray, 
    traj_2: np.ndarray
): 
  return np.abs(traj_1[-1] - traj_2[-1])


def evaluate(
    traj_1: np.ndarray, 
    traj_2: np.ndarray, 
    metrics: str = 'ADE'
) -> float:
  assert metrics in ('ADE', 'FDE'), "metrics should be ADE or FDE"
  return ADE(traj_1, traj_2) if metrics is 'ADE' else FDE(traj_1, traj_2)