if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import click
import pathlib
from diffusion_policy.common.replay_buffer import ReplayBuffer


@click.command()
@click.option('-i', '--input', required=True)
@click.option('-o', '--output', required=True)
@click.option('-t', '--target_eef_idx', default=8, type=int)
def main(input, output, target_eef_idx):
    buffer = ReplayBuffer.copy_from_path(input)
    obs = buffer['obs']
    action = buffer['action']
    prev_eef_target = obs[:,target_eef_idx:target_eef_idx+action.shape[1]]
    next_eef_target = prev_eef_target + action
    action[:] = next_eef_target
    buffer.save_to_path(zarr_path=output, chunk_length=-1)

if __name__ == '__main__':
    main()
