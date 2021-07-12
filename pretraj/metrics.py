import numpy as np


def ADE(
    traj_1: np.ndarray, 
    traj_2: np.ndarray
) -> float:
  return np.average(np.abs((traj_1 - traj_2)))


def FDE(
    traj_1: np.ndarray, 
    traj_2: np.ndarray
) -> float: 
  return np.abs(traj_1[-1] - traj_2[-1])