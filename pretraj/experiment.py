import json
import numpy as np
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Text, Dict
from pretraj.vehicle import Vehicle
from pretraj import predict
from pretraj.merics import ADE, FDE
from pretraj import *

pair_info = None
number_vehicles = None
vehicle_pairs_list = None
models_list = None
model_names = None

def warm_up(): 
  """Initialize fixed parameters for all experiments."""
  global pair_info, number_vehicles, vehicle_pairs_list, models_list, model_names

  number_vehicles = 150

  models_list = ['constantv', 'IDM', 'adapt', 'regularized_adapt', 'nn']
  model_names = {
      'constantv': 'constant velocity', 
      'IDM': 'IDM', 
      'adapt': 'adapt', 
      'regularized_adapt': 'regularized adapt',
      'nn': 'neural network'
  }

  with open(REDUCED_NGSIM_JSON_PATH) as fp:
      pair_info = json.load(fp)

  vehicle_pairs_list = \
    [(Vehicle(**pair_info[i]['ego']),
      Vehicle(**pair_info[i]['pre']))
      for i in range(number_vehicles)]


def experiment(
    observe_frames: list, 
    predict_frames: list, 
) -> Dict[Text,Dict[Text, int]]: # model_name -> ADE | FDE | hard_braking -> error
  """experiment."""
  result = {}
  for i, (ego, pre) in enumerate(vehicle_pairs_list):

    if i in outliers: continue

    if (i+1) % 10 == 0:
      print(f'{i+1}/{number_vehicles}')

    groundtruth_record = ego.space_headway_vector[observe_frames:observe_frames+predict_frames]

    for model in models_list:
      hard_braking, (ds_record, _, _) = predict.predict(ego, pre, observe_frames, predict_frames, model)
      if model not in result:
        result[model] = {'ADE':0, 'FDE':0, 'hard_braking':0}
      result[model]['ADE'] += \
          ADE(np.array(ds_record), np.array(groundtruth_record)) / number_vehicles / 3.2808399
      result[model]['FDE'] += \
          FDE(np.array(ds_record), np.array(groundtruth_record)) / number_vehicles / 3.2808399
      result[model]['hard_braking'] += hard_braking
    
  return result


def experiment_fixed_observe(draw_only=False):
  """fixed observe_frames, and change predict_frames."""
  observe_frames = 100
  predict_frames_list = [20, 40, 60, 80, 100]
  if not draw_only: 
    print('='*10 + 'fixed observe frames' + '='*10)
    results_list = []
    for predict_frames in predict_frames_list:
      print('predict frames:', predict_frames)
      result = experiment(observe_frames, predict_frames)
      results_list.append(result)
    
    Path(RESULT_DIR).mkdir(exist_ok=True)
    with open(FIXED_OBSERVE_JSON_PATH, 'w') as fp:
      json.dump(results_list, fp, indent=2)

  # draw figure
  if draw_only: 
    with open(FIXED_OBSERVE_JSON_PATH) as fp:
      results_list = json.load(fp)
  
  ADE_dict = {model: [result[model]['ADE'] 
      for result in results_list] for model in models_list}

  for model, ADE_list in ADE_dict.items():
    plt.plot(predict_frames_list, ADE_list, label=model_names[model])

  plt.xticks(predict_frames_list)
  plt.xlabel('Predict frames (10ms)')
  plt.ylabel('ADE (m)')
  plt.legend()
  plt.title('Evaluation of prediction models with 10s observe frames.')
  plt.savefig(FIXED_OBSERVE_FIG_PATH, dpi=400)
  plt.close()


def experiment_fixed_predict(draw_only=False):
  """fixed predict_frames, and change observe_frames."""
  # observe_frames_list = [20, 40, 60, 80, 100]
  observe_frames_list = np.arange(0, 100, 10) + 10
  predict_frames = 50
  if not draw_only: 
    print('='*10 + 'fixed predict frames' + '='*10)
    results_list = []
    for observe_frames in observe_frames_list:
      print('observe frames:', observe_frames)
      result = experiment(observe_frames, predict_frames)
      results_list.append(result)
    
    Path(RESULT_DIR).mkdir(exist_ok=True)
    with open(FIXED_PREDICT_JSON_PATH, 'w') as fp:
      json.dump(results_list, fp, indent=2)

  # draw figure
  if draw_only: 
    with open(FIXED_PREDICT_JSON_PATH) as fp:
      results_list = json.load(fp)
  
  ADE_dict = {model: [result[model]['ADE'] 
      for result in results_list] for model in models_list}

  for model, ADE_list in ADE_dict.items():
    plt.plot(observe_frames_list, ADE_list, label=model_names[model])

  plt.xticks(observe_frames_list)
  plt.xlabel('Observe frames (10ms)')
  plt.ylabel('ADE (m)')
  plt.legend()
  plt.title('Evaluation of prediction models with 5s predict frames.')
  plt.savefig(FIXED_PREDICT_FIG_PATH, dpi=400)
  plt.close()


def experiment_runtime():
  runtime_result = {}
  ego, pre = vehicle_pairs_list[0]
  observe_frames = 100
  predict_frames = 50
  for model in models_list:
    start_time = time.time()
    predict.predict(ego, pre, observe_frames, predict_frames, model)
    runtime_result[model] = time.time() - start_time
  with open(RUNTIME_PATH, 'w') as fp:
    json.dump(runtime_result, fp, indent=2)


if __name__ == '__main__':
  warm_up()
  experiment_fixed_observe(draw_only=False)
  experiment_fixed_predict(draw_only=False)
  experiment_runtime()