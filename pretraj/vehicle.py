from collections import namedtuple
import numpy as np

# Vehicle = namedtuple('Vehicle', ['vehicle_id', 'frame_id', 'vehicle_length', 'acc_vector', 'vel_vector', 'space_headway_vector'])

class Vehicle(object):
  def __init__(self, vehicle_id, frame_id, vehicle_length, acc_vector, vel_vector, space_headway_vector):
    self.vehicle_id = vehicle_id
    self.frame_id=frame_id
    self.vehicle_length=vehicle_length
    self.acc_vector=np.array(acc_vector)
    self.vel_vector=np.array(vel_vector)
    self.space_headway_vector=np.array(space_headway_vector)
