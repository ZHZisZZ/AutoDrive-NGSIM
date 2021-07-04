import numpy as np


class State(object):
  def __init__(self, a, v, ds):
    self.a = a; self.v = v; self.ds = ds


class Vehicle(object):
  def __init__(self, vehicle_id, frame_id, vehicle_length, acc_vector, vel_vector, space_headway_vector):
    self.vehicle_id = vehicle_id
    self.frame_id=frame_id
    self.vehicle_length=vehicle_length
    self.acc_vector=np.array(acc_vector)
    self.vel_vector=np.array(vel_vector)
    self.space_headway_vector=np.array(space_headway_vector)

  def state(self, i):
    return State(
        self.acc_vector[i],
        self.vel_vector[i],
        self.space_headway_vector[i])

  def states(self, observe_frames, predict_frames):
    return [self.state(i)
        for i in range(observe_frames, observe_frames+predict_frames)]
