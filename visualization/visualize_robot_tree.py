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

poses = z['data/action'][:]
qpos = z['data/robot_joint'][:]
ep_len = z['meta/episode_ends'][:]
ts = z['data/timestamp'][:]

dh_params = np.array(
[
    [0.1625, 0.0, 0.5 * pi, 0.0],
    [0.0, -0.425, 0, 0.0],
    [0.0, -0.3922, 0, 0.0],
    [0.1333, 0.0, 0.5 * pi, 0.0],
    [0.0997, 0.0, -0.5 * pi, 0.0],
    [0.0996, 0.0, 0.0, 0.0],
])

print(f"Visualizing Datset: {dataset_dir}")
# Determine start and end index of the episode
episode_index = int(input("Enter episode index to visualize: "))
start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
end_index = ep_len[episode_index]


viz_type = 'no'
robot = RobotSerial(dh_params)

fig = plt.figure(figsize=(64, 64))  # Create a figure
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

ax_robot =  plt.subplot(gs[0, 0], projection="3d")


plt.ion() 
print("Episode length is " , (end_index-start_index)," steps")
try:
    for i in range(start_index, end_index):
        # plt.clf()  # Clear the previous figure
        ax_robot.clear();a = np.arange(10)
        if viz_type == 'action':
            print("qpos for the episode is ", qpos[i])
            xyz = qpos[i][:3].reshape((3,1))
            rpy = qpos[i][3:]
            rotation_object = st.Rotation.from_rotvec(rpy)
            euler_angles = rotation_object.as_euler('zxy')  # Adjust 'zyx' as necessary based on your convention
            end = Frame.from_euler_3(euler_angles, xyz)  # Assuming this function expects Euler angles and position
            robot.inverse(end)
            robot.ax= ax_robot
            robot.draw()
            ax_robot.set_title('Robot Kinematics')
        else:
            robot.forward(qpos[i])
            # print(qpos[i])
            robot.ax= ax_robot
            robot.draw()
            ax_robot.set_title('Robot Kinematics')


        # plt.tight_layout()  # Ensure proper spacing between subplots
        plt.pause(0.0001)
        fig.canvas.flush_events()

except KeyboardInterrupt:
    plt.ioff()  # Turn off interactive mode
    plt.close()

plt.ioff()  # Turn off interactive mode
plt.close()








