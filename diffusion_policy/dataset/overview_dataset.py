import zarr 

root = zarr.open("data/pusht/pusht_cchi_v7_replay.zarr", mode="r")

print(root.tree())

for name, arr in root['data'].items():
  print(name, arr.shape, arr.dtype)
  if len(arr.shape) <= 3:
    print(f"{name}, data[0]: {arr[0]}")
  debug = 1

#  ├── data
#  │   ├── action (25650, 2) float32
#  │   ├── img (25650, 96, 96, 3) float32
#  │   ├── keypoint (25650, 9, 2) float32
#  │   ├── n_contacts (25650, 1) float32
#  │   └── state (25650, 5) float32
#  └── meta
#      └── episode_ends (206,) int64
# action (25650, 2) float32
# img (25650, 96, 96, 3) float32
# keypoint (25650, 9, 2) float32
# n_contacts (25650, 1) float32
# state (25650, 5) float32