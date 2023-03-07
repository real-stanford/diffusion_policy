if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import click
import pathlib
import h5py
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation

def read_all_actions(hdf5_file, metric_skip_steps=1):
    n_demos = len(hdf5_file['data'])
    all_actions = list()
    for i in tqdm(range(n_demos)):
        actions = hdf5_file[f'data/demo_{i}/actions'][:]
        all_actions.append(actions[metric_skip_steps:])
    all_actions = np.concatenate(all_actions, axis=0)
    return all_actions


@click.command()
@click.option('-i', '--input', required=True, help='input hdf5 path')
@click.option('-o', '--output', required=True, help='output hdf5 path. Parent directory must exist')
def main(input, output):
    # process inputs
    input = pathlib.Path(input).expanduser()
    assert input.is_file()
    output = pathlib.Path(output).expanduser()
    assert output.is_file()

    input_file = h5py.File(str(input), 'r')
    output_file = h5py.File(str(output), 'r')

    input_all_actions = read_all_actions(input_file)
    output_all_actions = read_all_actions(output_file)
    pos_dist = np.linalg.norm(input_all_actions[:,:3] - output_all_actions[:,:3], axis=-1)
    rot_dist = (Rotation.from_rotvec(input_all_actions[:,3:6]
        ) * Rotation.from_rotvec(output_all_actions[:,3:6]).inv()
        ).magnitude()

    print(f'max pos dist: {pos_dist.max()}')
    print(f'max rot dist: {rot_dist.max()}')

if __name__ == "__main__":
    main()
