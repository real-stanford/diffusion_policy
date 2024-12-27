import zarr
import numpy as np
import matplotlib.pyplot as plt
from viz_constants import VIZ_DIR

dataset_dir = VIZ_DIR
# Load the Zarr dataset
z = zarr.open(f'/home/bmv/diffusion_policy_mod_apr24/data/{dataset_dir}/replay_buffer.zarr', mode='r')

ep_len = z['meta/episode_ends'][:]
ft = z['data/ft_data'][:]
print("ft data shape:", ft.shape)
def plot_per_episode_ft(ax, episode_data):
    # Plot ft data for a specific episode on a given axis (subplot)
    ax.plot(episode_data)
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
    plt.show()

# Function to plot ft data for a particular episode
def plot_episode_ft(ft_data, ep_len, episode_index):
    # Plot ft data for a specific episode
    start_index = 0 if episode_index == 0 else ep_len[episode_index - 1]
    end_index = ep_len[episode_index]
    episode_ft_data = ft_data[start_index:end_index]
    with np.printoptions(threshold=np.inf):
        print("episode_ft_data:", episode_ft_data)
    plt.figure(figsize=(10, 6))
    plt.plot(episode_ft_data)
    plt.title(f'Force/Torque Data for Episode {episode_index}')
    plt.xlabel('Time Step')
    plt.ylabel('Force/Torque')
    plt.legend(['FX', 'FY', 'FZ', 'TX', 'TY', 'TZ'])  # Assuming order of dimensions
    plt.grid(True)
    plt.show()

print(f"Visualizing Datset: {dataset_dir}")
episode_index = int(input("Enter episode index to visualize || -1 for all data: "))
if episode_index>=0:
    plot_episode_ft(ft, ep_len, episode_index=episode_index)
elif episode_index==-1:
    visualize_all_episodes_ft(ft, ep_len)
else:
    raise NotImplementedError