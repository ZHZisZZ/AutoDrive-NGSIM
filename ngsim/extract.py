import pandas as pd
import numpy as np

NGSIM_PATH = 'NGSIM.csv'
TEMP_REDUCED_NGSIM_PATH = 'TEMP_REDUCED_NGSIM.csv'
REDUCED_NGSIM_PATH = 'REDUCED_NGSIM.csv'

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
  # df = pd.read_csv(TEMP_REDUCED_NGSIM_PATH)
  df = pd.read_csv('test.csv')
  # pair of to space_headway variance
  variance_dict = {}
  # get a list of vehicle_id
  ego_id_list = df.drop_duplicates(subset=['Vehicle_ID'])['Vehicle_ID']
  for ego_id in ego_id_list:
    # start frame_id with preceding car does not change for 48 frames
    df_ego = df[df['Vehicle_ID']==ego_id]
    frames_cnt = 0
    flag = False
    frame_id = df_ego['Frame_ID'][0] 
    frame_index = 0
    last_pre_id = df_ego['Preceding'][0]
    last_lane_id = df_ego['Lane_ID'][0]

    for index, row in df_ego.iterrows():
      if row['Preceding'] != last_pre_id or \
         row['Preceding'] == 0.0 or \
         row['Space_Headway'] == 0.0 or \
         row['Lane_ID'] != last_lane_id:
         frame_index = index
         frame_id = row['Frame_ID']
         frames_cnt = 1
         last_pre_id = row['Preceding']
      else:
        frames_cnt += 1
        if frames_cnt >= NUM_FRAMES:
          flag = True
          break

    
    print(flag)
    print(frame_id)
    # to get index from frame_id
    # print(df_ego[df_ego['Frame_ID']==frame_id].index[0])
    # print(frame_index)

    variance_dict[(ego_id, df_ego[df_ego['Frame_ID']==frame_id]['Preceding'][0], frame_id)] = \
      df_ego[frame_index:frame_index+NUM_FRAMES]['Space_Headway'].to_numpy()
    print(variance_dict)


  df.to_csv(REDUCED_NGSIM_PATH, index=False)
  

if __name__ == '__main__':
  filter_one()

  # filter_two()