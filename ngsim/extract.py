import pandas as pd
import numpy as np
import json

NGSIM_PATH = 'NGSIM.csv'
TEMP_REDUCED_NGSIM_PATH = 'TEMP_REDUCED_NGSIM.csv'
REDUCED_NGSIM_PATH = 'REDUCED_NGSIM.csv'
REDUCED_NGSIM_JSON_PATH = 'REDUCED_NGSIM.json'

NUM_FRAMES = 48
MAX_PAIRS = 1000

def filter_one():
  """reduce the size of data_file."""
  # read NGSIM dataset
  df = pd.read_csv(NGSIM_PATH)
  # remove duplicated rows
  df = df.drop_duplicates(subset=['Vehicle_ID', 'Frame_ID'])
  # remove other locations other than 'us-101'
  df = df[df['Location'] == 'us-101']
  # sort according to 'Vehicle_ID' and 'Frame_ID'
  df = df.sort_values(['Vehicle_ID', 'Frame_ID'])
  # retain the information we need and discard others
  df = df.loc[:, ['Vehicle_ID', 'Frame_ID', 'v_length', 'v_Class', 'v_Vel', 'v_Acc', 
                  'Lane_ID', 'Preceding', 'Following', 'Space_Headway', 'Time_Headway']]
  # write to an intermediate file
  df.to_csv(TEMP_REDUCED_NGSIM_PATH, index=False)


def filter_two():
  """only keep ego_pre car pairs with untrivial relations."""
  df = pd.read_csv(TEMP_REDUCED_NGSIM_PATH)
  # df = pd.read_csv('test.csv')
  # pair of to space_headway variance
  variance_dict = {}
  # get a list of vehicle_id
  ego_id_list = df.drop_duplicates(subset=['Vehicle_ID'])['Vehicle_ID']
  for ego_id in ego_id_list:
    print(ego_id)
    df_ego = df[df['Vehicle_ID']==ego_id].reset_index(drop=True)
    frame_id = df_ego['Frame_ID'].iloc[0]
    last_pre_id = df_ego['Preceding'].iloc[0]
    last_lane_id = df_ego['Lane_ID'].iloc[0]
    last_frame_id = df_ego['Frame_ID'].iloc[0]
    frames_cnt = 0
    flag = False

    for _, row in df_ego.iterrows():
      # df_pre = df[df['Vehicle_ID']==row['Preceding']].reset_index(drop=True)
      if   row['Preceding'] != last_pre_id or \
           row['Lane_ID'] != last_lane_id or \
           row['Frame_ID'] != last_frame_id + 1 or \
           row['Preceding'] == 0.0 or \
           row['Space_Headway'] == 0.0:
          #  not any(df_pre['Frame_ID']==row['Frame_ID']):
         frame_id = row['Frame_ID']
         frames_cnt = 1
         last_pre_id = row['Preceding']
      else:
        frames_cnt += 1
        if frames_cnt >= NUM_FRAMES:
          flag = True
          break
      last_frame_id = row['Frame_ID']

    if flag:
      df_ego = df_ego[(df_ego['Frame_ID']>=frame_id) & (df_ego['Frame_ID']<frame_id+NUM_FRAMES)].reset_index(drop=True)
      pre_id = df_ego['Preceding'].iloc[0]
      # TODO
      df_pre = (df['Vehicle_ID']==pre_id) & (df['Frame_ID']>=frame_id) & (df['Frame_ID']<frame_id+NUM_FRAMES)
      if sum(df_pre) < NUM_FRAMES: continue
      #
      traj = df_ego['Space_Headway'].to_numpy()
      variance_dict[(ego_id, pre_id, int(frame_id))] = np.var(traj)


  sorted_pairs = sorted(variance_dict, key=variance_dict.__getitem__, reverse=False)
  print(sorted_pairs)
  print('len:', len(sorted_pairs))

  new_df = None
  output_dict = []
  for pair in sorted_pairs:
    ego_id, pre_id, frame_id = pair
    df_ego = df[(df['Vehicle_ID']==ego_id) & (df['Frame_ID']>=frame_id) & (df['Frame_ID']<frame_id+NUM_FRAMES)].reset_index(drop=True)
    # print(df_ego)
    new_df = pd.concat([new_df, df_ego], axis=0)
    ego_info = {
        'vehicle_id': int(ego_id),
        'frame_id': int(frame_id), 
        'vehicle_length': int(df_ego['v_length'].iloc[0]),
        'acc_vector': df_ego['v_Acc'].tolist(), 
        'vel_vector': df_ego['v_Vel'].tolist(), 
        'space_headway_vector': df_ego['Space_Headway'].tolist()}
    df_pre = df[(df['Vehicle_ID']==pre_id) & (df['Frame_ID']>=frame_id) & (df['Frame_ID']<frame_id+NUM_FRAMES)].reset_index(drop=True)
    new_df = pd.concat([new_df, df_pre], axis=0)
    # print(df_pre)
    pre_info = {
        'vehicle_id': int(pre_id),
        'frame_id': int(frame_id), 
        'vehicle_length': int(df_pre['v_length'].iloc[0]), 
        'acc_vector': df_pre['v_Acc'].tolist(), 
        'vel_vector': df_pre['v_Vel'].tolist(), 
        'space_headway_vector': df_pre['Space_Headway'].tolist()}
    output_dict.append({'ego':ego_info, 'pre':pre_info})


  with open(REDUCED_NGSIM_JSON_PATH, 'w') as out_json:
    json.dump(output_dict, out_json, indent=2)

  new_df.to_csv(REDUCED_NGSIM_PATH, index=False)
  

if __name__ == '__main__':
  # filter_one()

  filter_two()