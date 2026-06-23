import zarr 
import json 
import numpy as np
import pandas as pd

from io import BytesIO
from PIL import Image
from pathlib import Path

parquet_dir = Path("data/chemdroid-insertion-dataset")
out_path = "data/chemdroid-insertion-zarr"

dfs = []
episode_ends = []

# Calculate rows 
n_rows = 0
column_names = []
first_file = next(parquet_dir.glob("**/*.parquet"))
file_data = pd.read_parquet(first_file)
column_names = file_data.columns.tolist()
for pq in sorted(parquet_dir.glob("**/*.parquet")):
  df = pd.read_parquet(pq)
  n_rows += df.shape[0]

# Create columns of zarr
root = zarr.open(out_path, mode="w")
data = root.create_group("data")
meta = root.create_group("meta")

dims = dict()
data_dict = dict()
for col in column_names:
  dt = col
  if type(file_data[col][0]).__name__ == 'dict':
    column_data = []
    for img_name in file_data[col].iloc:
      img = Image.open(BytesIO(img_name['bytes']))
      column_data.append(img)
    column_data = np.stack(column_data)
  else: 
    column_data = np.stack(file_data[col].to_numpy())
  if '.' in dt:
    dt = dt.replace('.', '_')
  if dt == "actions":
    dt = "action"
  dims[col] = column_data.shape[1:]
  if len(column_data.shape) <=3: 
    chunk_size = 1024
  else:
    chunk_size = 64
  data_dict[col] = root.create_dataset(dt,
                                     shape=(n_rows, *column_data.shape[1:]),
                                     chunks=(chunk_size, *column_data.shape[1:]),
                                     dtype=column_data.dtype)
del file_data
offset = 0
for pq in sorted(parquet_dir.glob("**/*.parquet")):
  print(f"Processing file {pq.name}")
  df = pd.read_parquet(pq)
  for dt in column_names:
    # print(f" {dt} of type {type(df[dt][0]).__name__}")
    if type(df[dt][0]).__name__ == 'dict':
      column_data = []
      for img_name in df[dt].iloc:
        img = Image.open(BytesIO(img_name['bytes']))
        column_data.append(img)
      column_data = np.stack(column_data)
    else: 
      column_data = np.stack(df[dt].to_numpy())
    data_dict[dt][offset:offset+df.shape[0]] = column_data
    # print(f"column shape of {dt} is {dt_list[-1].shape}")
  offset += df.shape[0]
  episode_ends.append(offset)

# for i, dt in enumerate(column_list):
#   tmp_data = np.concatenate([x[i] for x in dfs], axis=0)
#   if len(tmp_data.shape) <=3: 
#     chunk_size = 1024
#   else:
#     chunk_size = 64
#   if '.' in dt:
#     dt = dt.replace('.', '_')
#   if dt == "actions":
#     dt = "action"
#   data.create_dataset(dt, data=tmp_data, chunks=(chunk_size, *tmp_data.shape[1:]))

meta.create_dataset("episode_ends", data=np.array(episode_ends, dtype=np.int64))

# (['joint_position', 'joint_velocity', 'end_effector_pose',
#        'gripper_position', 'actions', 'timestamp', 'frame_index',
#        'episode_index', 'index', 'task_index', 'base_0_rgb.bytes',
#        'base_0_rgb.path', 'left_wrist_0_rgb.bytes', 'left_wrist_0_rgb.path',
#        'right_wrist_0_rgb.bytes', 'right_wrist_0_rgb.path'],
#       dtype='object')