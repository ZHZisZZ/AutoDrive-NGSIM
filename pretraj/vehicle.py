import pretraj
import numpy as np


class State(object):
  def __init__(self, ds=0, v=0, a=0):
    self.ds = ds; self.v = v; self.a = a


class Vehicle(object):
  def __init__(self, vehicle_id, frame_id, 
               vehicle_length, vel_vector, 
               space_headway_vector, **argvs):
    self.vehicle_id = vehicle_id
    self.frame_id=frame_id
    self.vehicle_length=vehicle_length
    self.vel_vector=np.array(vel_vector)
    self.acc_vector=np.diff(self.vel_vector, append=0) / .1
    self.space_headway_vector=np.array(space_headway_vector)

  def state(self, i):
    return State(
        ds=self.space_headway_vector[i],
        v=self.vel_vector[i],
        a=self.acc_vector[i])

  def states(self, observe_frames, predict_frames=None):
    if not predict_frames: 
      predict_frames = pretraj.NUM_FRAMES - observe_frames
    return [self.state(i)
        for i in range(observe_frames, observe_frames+predict_frames)]
