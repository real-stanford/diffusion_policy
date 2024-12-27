import zarr
import numpy as np
import matplotlib.pyplot as plt
from viz_constants import VIZ_DIR


dataset_dir = VIZ_DIR
# Load the Zarr dataset
z = zarr.open(f'/home/bmv/diffusion_policy_mod_apr24/data/{dataset_dir}/replay_buffer.zarr', mode='r')

ep_len = z['meta/episode_ends'][:]
ep_len = np.diff(np.insert(ep_len,0,0))

plt.figure(figsize=(10, 6))
plt.plot(ep_len)
plt.title('Episode Lengths')
plt.xlabel('Episode Number')
plt.ylabel('Length')
plt.grid(True)
plt.show()


