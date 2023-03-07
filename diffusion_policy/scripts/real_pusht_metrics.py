if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import click
import av
import cv2
import collections
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import threadpoolctl
from matplotlib import pyplot as plt
import json

def get_t_mask(img, hsv_ranges=None):
    if hsv_ranges is None:
        hsv_ranges = [
            [0,255],
            [130,216],
            [150,230]
        ]
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = np.ones(img.shape[:2], dtype=bool)
    for c in range(len(hsv_ranges)):
        l, h = hsv_ranges[c]
        mask &= (l <= hsv_img[...,c])
        mask &= (hsv_img[...,c] <= h)
    return mask

def get_mask_metrics(target_mask, mask):
    total = np.sum(target_mask)
    i = np.sum(target_mask & mask)
    u = np.sum(target_mask | mask)
    iou = i / u
    coverage = i / total
    result = {
        'iou': iou,
        'coverage': coverage
    }
    return result

def get_video_metrics(video_path, target_mask, use_tqdm=True):
    threadpoolctl.threadpool_limits(1)
    cv2.setNumThreads(1)

    metrics = collections.defaultdict(list)
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        iterator = None
        if use_tqdm:
            iterator = tqdm(container.decode(stream), total=stream.frames)
        else:
            iterator = container.decode(stream)
        for frame in iterator:
            img = frame.to_ndarray(format='rgb24')
            mask = get_t_mask(img)
            metric = get_mask_metrics(
                target_mask=target_mask, mask=mask)
            for k, v in metric.items():
                metrics[k].append(v)
    return metrics

def worker(x):
    return get_video_metrics(*x)

@click.command()
@click.option(
    '--reference', '-r', required=True, 
    help="Reference video whose last frame will define goal.")
@click.option(
    '--input', '-i', required=True,
    help='Dataset path to evaluate.')
@click.option(
    '--camera_idx', '-ci', default=0, type=int,
    help="Camera index to compute metrics")
@click.option('--n_workers', '-n', default=20, type=int)
def main(reference, input, camera_idx, n_workers):
    # read last frame of the reference video to get target mask
    last_frame = None
    with av.open(reference) as container:
        stream = container.streams.video[0]
        for frame in tqdm(
                container.decode(stream), 
                total=stream.frames):
            last_frame = frame

    last_img = last_frame.to_ndarray(format='rgb24')
    target_mask = get_t_mask(last_img)

    # path = '/home/ubuntu/dev/diffusion_policy/data/pusht_real/eval_20230109/diffusion_hybrid_ep136/videos/4/0.mp4'
    # last_frame = None
    # with av.open(path) as container:
    #     stream = container.streams.video[0]
    #     for frame in tqdm(
    #             container.decode(stream), 
    #             total=stream.frames):
    #         last_frame = frame
    # img = last_frame.to_ndarray(format='rgb24')
    # mask = get_t_mask(img)

    # get metrics for each episode
    episode_video_path_map = dict()
    input_dir = pathlib.Path(input)
    input_video_dir = input_dir.joinpath('videos')
    for vid_dir in input_video_dir.glob("*/"):
        episode_idx = int(vid_dir.stem)
        video_path = vid_dir.joinpath(f'{camera_idx}.mp4')
        if video_path.exists():
            episode_video_path_map[episode_idx] = str(video_path.absolute())

    episode_idxs = sorted(episode_video_path_map.keys())
    print(f"Found video for following episodes: {episode_idxs}")

    # run
    with mp.Pool(n_workers) as pool:
        args = list()
        for idx in episode_idxs:
            args.append((episode_video_path_map[idx], target_mask))
        results = pool.map(worker, args)
    episode_metric_map = dict()
    for idx, result in zip(episode_idxs, results):
        episode_metric_map[idx] = result

    # aggregate metrics
    agg_map = collections.defaultdict(list)
    for idx, metric in episode_metric_map.items():
        for key, value in metric.items():
            agg_map['max/'+key].append(np.max(value))
            agg_map['last/'+key].append(value[-1])

    final_metric = dict()
    for key, value in agg_map.items():
        final_metric[key] = np.mean(value)

    # save metrics
    print('Saving metrics!')
    with input_dir.joinpath('metrics_agg.json').open('w') as f:
        json.dump(final_metric, f, sort_keys=True, indent=2)
    
    with input_dir.joinpath('metrics_raw.json').open('w') as f:
        json.dump(episode_metric_map, f, sort_keys=True, indent=2)
    print('Done!')

if __name__ == '__main__':
    main()
