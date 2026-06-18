import zarr 
import numpy as np
import pandas as pd

from io import BytesIO
from PIL import Image
from pathlib import Path

parquet_dir = Path("data/chemdroid-insertion-dataset")
out_path = "data/chemdroid-insertion-zarr"

dfs = []
episode_ends = []
count = 0

# Get names of columns in first parquet file
column_list = []
for pq in sorted(parquet_dir.glob("**/*.parquet")):
  df = pd.read_parquet(pq)
  column_list = df.columns.tolist()
  break

for pq in sorted(parquet_dir.glob("**/*.parquet")):
  df = pd.read_parquet(pq)
  dt_list = []
  for dt in column_list:
    if type(df[dt][0]).__name__ == 'bytes':
      column_data = []
      for img_name in df[dt].iloc:
        img = Image.open(BytesIO(img_name))
        column_data.append(img)
      column_data = np.stack(column_data)
    else: 
      column_data = np.stack(df[dt].to_numpy())
    dt_list.append(column_data)
    print(f"column shape of {dt} is {dt_list[-1].shape}")
  dfs.append(tuple(dt_list))
  count += len(df)
  episode_ends.append(count)
  break

root = zarr.open(out_path, mode="w")
data = root.create_group("data")
meta = root.create_group("meta")

for i, dt in enumerate(column_list):
  tmp_data = np.concatenate([x[i] for x in dfs], axis=0)
  if len(tmp_data.shape) <=3: 
    chunk_size = 1024
  else:
    chunk_size = 64
  data.create_dataset(dt, data=tmp_data, chunks=(chunk_size, *tmp_data.shape[1:]))

meta.create_dataset("episode_ends", data=np.array(episode_ends, dtype=np.int64))

# (['joint_position', 'joint_velocity', 'end_effector_pose',
#        'gripper_position', 'actions', 'timestamp', 'frame_index',
#        'episode_index', 'index', 'task_index', 'base_0_rgb.bytes',
#        'base_0_rgb.path', 'left_wrist_0_rgb.bytes', 'left_wrist_0_rgb.path',
#        'right_wrist_0_rgb.bytes', 'right_wrist_0_rgb.path'],
#       dtype='object')