if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import click
import pathlib
import zarr
import cv2
import threadpoolctl
from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer

@click.command()
@click.option('--input', '-i',  required=True)
@click.option('--output', '-o', default=None)
@click.option('--resolution', '-r', default='640x480')
@click.option('--n_decoding_threads', '-nd', default=-1, type=int)
@click.option('--n_encoding_threads', '-ne', default=-1, type=int)
def main(input, output, resolution, n_decoding_threads, n_encoding_threads):
    out_resolution = tuple(int(x) for x in resolution.split('x'))
    input = pathlib.Path(os.path.expanduser(input))
    in_zarr_path = input.joinpath('replay_buffer.zarr')
    in_video_dir = input.joinpath('videos')
    assert in_zarr_path.is_dir()
    assert in_video_dir.is_dir()
    if output is None:
        output = input.joinpath(resolution + '.zarr.zip')
    else:
        output = pathlib.Path(os.path.expanduser(output))

    if output.exists():
        click.confirm('Output path already exists! Overrite?', abort=True)

    cv2.setNumThreads(1)
    with threadpoolctl.threadpool_limits(1):
        replay_buffer = real_data_to_replay_buffer(
            dataset_path=str(input),
            out_resolutions=out_resolution,
            n_decoding_threads=n_decoding_threads,
            n_encoding_threads=n_encoding_threads
        )
    
    print('Saving to disk')
    if output.suffix == '.zip':
        with zarr.ZipStore(output) as zip_store:
            replay_buffer.save_to_store(
                store=zip_store
            )
    else:
        with zarr.DirectoryStore(output) as store:
            replay_buffer.save_to_store(
                store=store
            )

if __name__ == '__main__':
    main()
