import zarr
import numpy as np
import matplotlib.pyplot as plt
from viz_constants import VIZ_DIR

dataset_dir = VIZ_DIR
# Load the Zarr dataset
z = zarr.open(f'/home/bmv/diffusion_policy_mod_apr24/data/{dataset_dir}/replay_buffer.zarr', mode='r')

ep_len = z['meta/episode_ends'][:]
ft = z['data/ft_data'][:]
poses = z['data/replica_eef_pose'][:]
#print("ft data", ft)

def plot_per_episode_ft(ax, episode_data):
    # Plot ft data for a specific episode on a given axis (subplot)
    ax.plot(episode_data[:,1])
    ax.grid(True)

def visualize_all_episodes_ft(ft_data, ep_len, num_cols=5):
    num_episodes = len(ep_len)
    num_rows = (num_episodes + num_cols - 1) // num_cols  # Calculate number of rows based on number of episodes and columns
    print(f"Total number of episodes: {num_episodes}")

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    for episode_index in range(num_episodes):
        row = episode_index // num_cols
        col = episode_index % num_cols
        ax = axs[row, col] if num_rows > 1 else axs[col]
        
        start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
        end_index = ep_len[episode_index]
        episode_data = ft_data[start_index:end_index]
        
        plot_per_episode_ft(ax, episode_data)

    # Adjust layout and spacing
    fig.tight_layout(pad=0.5)
    # plt.show()

# Function to plot ft data for a particular episode
def plot_episode_ft(ft_data, ep_len, episode_index):
    # Plot ft data for a specific episode
    start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
    end_index = ep_len[episode_index]
    episode_ft_data = ft_data[start_index:end_index]
    reshaped_ft_data = episode_ft_data.reshape(-1, 6)
    print("reshaped:",len(reshaped_ft_data))
    # print("epusde_ft_data:", episode_ft_data)
    # print("epusdie len:", len(episode_ft_data))
    plt.figure(figsize=(10, 6))
    for i in range(reshaped_ft_data.shape[1]):
        plt.plot(reshaped_ft_data[:, i], label=f'Component {i+1}')
    plt.title(f'Force/Torque Data for Episode {episode_index}')
    plt.xlabel('Time Step')
    plt.ylabel('Force/Torque')
    plt.legend(['FX', 'FY', 'FZ', 'TX', 'TY', 'TZ']) 
    plt.grid(True)
    plt.show()

def plot_episode_ft_scatter(ft_data, ep_len, episode_index, set_index=0):
    # Determine the start and end indices for the specific episode
    start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
    end_index = ep_len[episode_index]
    
    # Extract only the 3rd column (FZ) data for the specific episode and specified set
    episode_ft_data_fz = ft_data[start_index:end_index, set_index, 2]  # 2 is the index for the 3rd column
    x = poses[:, 0][start_index:end_index]
    y = poses[:, 1][start_index:end_index]
    z = poses[:, 2][start_index:end_index]

    print("z values", z)

    for j in range(len(episode_ft_data_fz)):
        print(f"Episode FZ {j}: {episode_ft_data_fz[j]}")
        print(f"Episode X {j}: {x[j]}, Y {j}: {y[j]}, Z {j}: {z[j]}")


    # Create scatter plot for the FZ data
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(episode_ft_data_fz)), episode_ft_data_fz, label='FZ', color='b')
    
    # Set plot titles and labels
    plt.title(f'Scatter Plot of FZ Data for Episode {episode_index}, Set {set_index}')
    plt.xlabel('Time Step')
    plt.ylabel('FZ')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_episode_ft_scatter_frequency(ft_data, ep_len, episode_index, set_index=0):
    # Determine the start and end indices for the specific episode
    start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
    end_index = ep_len[episode_index]
    
    # Extract the specific episode data
    episode_ft_data = ft_data[start_index:end_index]
    reshaped_ft_data = episode_ft_data.reshape(-1, 6)
    
    # Create scatter plot for the reshaped FZ data
    plt.figure(figsize=(10, 6))
    for i in range(reshaped_ft_data.shape[1]):
        plt.scatter(range(reshaped_ft_data.shape[0]), reshaped_ft_data[:, i], label=f'Component {i+1}')
    plt.title(f'Scatter Plot of Force/Torque Data for Episode {episode_index}')
    plt.xlabel('Time Step')
    plt.ylabel('Force/Torque Value')
    plt.legend(['FX', 'FY', 'FZ', 'TX', 'TY', 'TZ'])
    plt.grid(True)
    plt.show()


# Function to plot all FX data in one plot
# def plot_all_fx_in_one_plot(ft_data, ep_len):
#     plt.figure(figsize=(15, 6))

#     for episode_index in range(len(ep_len)):
#         start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
#         end_index = ep_len[episode_index]
#         episode_data = ft_data[start_index:end_index]  # Only FX values
#         plt.plot(episode_data, alpha=0.5)  # Plot each episode's FX with some transparency

#     plt.title('Force in X direction (FX) for All Episodes')
#     plt.xlabel('Time Step')
#     plt.ylabel('Force (N)')
#     plt.grid(True)
#     plt.show()
print(f"Visualizing Datset: {dataset_dir}")
episode_index = int(input("Enter episode index to visualize || -1 for all data: "))
if episode_index>=0:
    plot_episode_ft_scatter_frequency(ft, ep_len, episode_index=episode_index)
elif episode_index==-1:
    plot_all_fx_in_one_plot(ft, ep_len)
else:
    raise NotImplementedError