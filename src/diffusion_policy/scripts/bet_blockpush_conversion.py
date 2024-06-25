if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)


import os
import click
import pathlib
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer

@click.command()
@click.option('-i', '--input', required=True, help='input dir contains npy files')
@click.option('-o', '--output', required=True, help='output zarr path')
@click.option('--abs_action', is_flag=True, default=False)
def main(input, output, abs_action):
    data_directory = pathlib.Path(input)
    observations = np.load(
        data_directory / "multimodal_push_observations.npy"
    )
    actions = np.load(data_directory / "multimodal_push_actions.npy")
    masks = np.load(data_directory / "multimodal_push_masks.npy")

    buffer = ReplayBuffer.create_empty_numpy()
    for i in range(len(masks)):
        eps_len = int(masks[i].sum())
        obs = observations[i,:eps_len].astype(np.float32)
        action = actions[i,:eps_len].astype(np.float32)
        if abs_action:
            prev_eef_target = obs[:,8:10]
            next_eef_target = prev_eef_target + action
            action = next_eef_target
        data = {                              
            'obs': obs,
            'action': action
        }
        buffer.add_episode(data)

    buffer.save_to_path(zarr_path=output, chunk_length=-1)

if __name__ == '__main__':
    main()
