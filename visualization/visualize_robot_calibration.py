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
z = zarr.open(f'/home/bmv/diffusion_policy_mod_apr24/data/{dataset_dir}/replay_buffer.zarr', mode='r')

ep_len = z['meta/episode_ends'][:]
robot_eef = z['data/robot_eef_pose'][:]
replica_eef = z['data/action'][:]

def plot_episode_ft(robot_eef, replica_eef, ep_len, episode_index):
    
    if episode_index == -1:

        # Select ft data for the episode
        ep_robot_eef = robot_eef[:]
        ep_replica_eef = replica_eef[:]
    
    else:
        # Determine start and end index of the episode
        start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
        end_index = ep_len[episode_index]

        # Select ft data for the episode
        ep_robot_eef = robot_eef[start_index:end_index]
        ep_replica_eef = replica_eef[start_index:end_index]

    # Calculate MSE for each component
    mse_error = np.mean((ep_robot_eef[:, 0:3] - ep_replica_eef[:, 0:3])**2, axis=0)*1000
    mae_error = np.mean(np.abs((ep_robot_eef[:, 0:3] - ep_replica_eef[:, 0:3])), axis=0)*1000
    print(f'Mean Squared Error milimeters: {mse_error}')
    print(f'Mean Absolute Error milimeters: {mae_error}')


    mse_error_rad = np.mean(np.abs(ep_robot_eef[:, 3:] - ep_replica_eef[:, 3:]), axis=0)
    mse_error_deg= np.rad2deg(mse_error_rad)
    print(f'Mean Absolute Error Radians: {mse_error_rad}')
    print(f'Mean Absolute Error Degrees: {mse_error_deg}')

    # Extract x, y, z coordinates for plotting
    robot_x = ep_robot_eef[:, 0]
    robot_y = ep_robot_eef[:, 1]
    robot_z = ep_robot_eef[:, 2]
    replica_x = ep_replica_eef[:, 0]
    replica_y = ep_replica_eef[:, 1]
    replica_z = ep_replica_eef[:, 2]

    # Plotting the trajectories
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot robot and replica trajectories
    ax.plot(robot_x, robot_y, robot_z, label='Robot EEF', color='b')
    ax.plot(replica_x, replica_y, replica_z, label='Replica EEF', color='r')

    # Set labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.title(f'Trajectories for Episode {episode_index}')
    plt.show()

# Example: Plotting episode 0
print(f"Visualizing Datset: {dataset_dir}")
episode_index = int(input("Enter episode index to visualize || -1 for all data: "))
plot_episode_ft(robot_eef, replica_eef, ep_len, episode_index)


