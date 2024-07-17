import zarr
import numpy as np
from visual_kinematics.RobotSerial import *
from visual_kinematics.RobotTrajectory import *
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Circle
from math import pi
import scipy.spatial.transform as st
from viz_constants import VIZ_DIR

dataset_dir = VIZ_DIR
# Load the Zarr dataset
z = zarr.open(f'/home/bmv/diffusion_policy_new/data/{dataset_dir}/replay_buffer.zarr', mode='r')

poses = z['data/robot_eef_pose'][:]
# qpos = z['data/robot_joint'][:]
ep_len = z['meta/episode_ends'][:]
ts = z['data/timestamp'][:]

# Determine start and end index of the episode
print(f"Visualizing Datset: {dataset_dir}")
episode_index = int(input("Enter episode index to visualize || -1 for all data: "))

# visualise all tyrajectories
if episode_index == -1:
    # Extract x, y, z coordinates
    x = poses[:, 0]
    y = poses[:, 1]
    z = poses[:, 2]

# visualise specific trajectory
else:
    start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
    end_index = ep_len[episode_index]

    x = poses[:, 0][start_index:end_index]
    y = poses[:, 1][start_index:end_index]
    z = poses[:, 2][start_index:end_index]

    print("poses in the episode are: ", poses[start_index:end_index])
    # print(z)
# Create 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)

# Set labels and title
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Robot Poses')

ax.set_xlim([-1, 1])  # Set limits for the x-axis
ax.set_ylim([-1, 1])  # Set limits for the y-axis
# ax.set_zlim([-1, 1])  # Set limits for the z-axis

plt.show()



