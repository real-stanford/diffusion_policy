import zarr

dataset_path = "data/replay_buffer.zarr"
zarr_file = zarr.open(dataset_path, mode='r')

print(f"Zarr Keys: {list(zarr_file.keys())}")
print(f"action{zarr_file['data']['action']}")