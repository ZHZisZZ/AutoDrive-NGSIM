from collections import namedtuple

TOTAL_FRAMES = 1000

Vehicle = namedtuple('Vehicle', ['vehicle_id', 'precede_id', 'frame_id', 'vehicle_length', 'acc_vector', 'vel_vector', 'space_headway_vector'])
# VehiclePair = namedtuple('VehiclePair', ['ego_vehicle', 'pre_vehicle'])