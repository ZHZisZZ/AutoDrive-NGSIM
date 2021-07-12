import json
import numpy as np
import json
import time
import os
from pathlib import Path
from typing import Text, Dict
import matplotlib.pyplot as plt
import pretraj
from pretraj import simulate
from pretraj import models
from pretraj import metrics


def experiment(
    observe_frames: list, 
    predict_frames: list, 
) -> Dict[Text,Dict[Text, int]]: # model_name -> ADE | FDE | hard_braking -> error
  """experiment."""
  result = {}
  for i, (ego, pre) in enumerate(pretraj.vehicle_pairs_list):

    if i in pretraj.outliers: continue

    if (i+1) % 10 == 0:
      print(f'{i+1}/{pretraj.NUM_PAIRS}')

    groundtruth_record = ego.space_headway_vector[observe_frames:observe_frames+predict_frames]

    for model in simulate.models_list:
      hard_braking, (ds_record, _, _) = simulate.simulate(ego, pre, observe_frames, predict_frames, model)
      if model not in result:
        result[model] = {'ADE':0, 'FDE':0, 'hard_braking':0}
      result[model]['ADE'] += \
          metrics.ADE(np.array(ds_record), np.array(groundtruth_record)) / pretraj.NUM_PAIRS / 3.2808399
      result[model]['FDE'] += \
          metrics.FDE(np.array(ds_record), np.array(groundtruth_record)) / pretraj.NUM_PAIRS / 3.2808399
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
    
    Path(pretraj.RESULT_DIR).mkdir(exist_ok=True)
    with open(pretraj.FIXED_OBSERVE_JSON_PATH, 'w') as fp:
      json.dump(results_list, fp, indent=2)

  # draw figure
  if draw_only: 
    with open(pretraj.FIXED_OBSERVE_JSON_PATH) as fp:
      results_list = json.load(fp)
  
  ADE_dict = {model: [result[model]['ADE'] 
      for result in results_list] for model in simulate.models_list}

  for model, ADE_list in ADE_dict.items():
    plt.plot(predict_frames_list, ADE_list, label=model)

  plt.xticks(predict_frames_list)
  plt.xlabel('Predict frames (0.1s)')
  plt.ylabel('ADE (m)')
  plt.legend()
  plt.title('Evaluation of prediction models with 10s observe frames')
  plt.savefig(pretraj.FIXED_OBSERVE_FIG_PATH, dpi=400)
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
    
    Path(pretraj.RESULT_DIR).mkdir(exist_ok=True)
    with open(pretraj.FIXED_PREDICT_JSON_PATH, 'w') as fp:
      json.dump(results_list, fp, indent=2)

  # draw figure
  if draw_only: 
    with open(pretraj.FIXED_PREDICT_JSON_PATH) as fp:
      results_list = json.load(fp)
  
  ADE_dict = {model: [result[model]['ADE'] 
      for result in results_list] for model in simulate.models_list}

  for model, ADE_list in ADE_dict.items():
    plt.plot(observe_frames_list, ADE_list, label=model)

  plt.xticks(observe_frames_list)
  plt.xlabel('Observe frames (0.1s)')
  plt.ylabel('ADE (m)')
  plt.legend()
  plt.title('Evaluation of prediction models with 5s predict frames')
  plt.savefig(pretraj.FIXED_PREDICT_FIG_PATH, dpi=400)
  plt.close()


def experiment_runtime():
  runtime_result = {}
  ego, pre = pretraj.vehicle_pairs_list[0]
  observe_frames = 100
  predict_frames = 50
  for model in simulate.models_list:
    start_time = time.time()
    simulate.simulate(ego, pre, observe_frames, predict_frames, model)
    runtime_result[model] = time.time() - start_time
  with open(pretraj.RUNTIME_PATH, 'w') as fp:
    json.dump(runtime_result, fp, indent=2)


if __name__ == '__main__':
  # if 'checkpoint' not in os.listdir('pretraj') or \
  #    'pretrain_model.pt' not in os.listdir(pretraj.CHECKPOINT_DIR):
  models.pretrain_neural_network()
  # experiment_fixed_observe(draw_only=False)
  experiment_fixed_predict(draw_only=False)
  experiment_runtime()