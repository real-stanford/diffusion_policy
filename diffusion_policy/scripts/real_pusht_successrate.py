if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import click
import collections
import numpy as np
from tqdm import tqdm
import json

@click.command()
@click.option(
    '--reference', '-r', required=True,
    help='Reference metrics_raw.json from demonstration dataset.'
)
@click.option(
    '--input', '-i', required=True,
    help='Data search path'
)
def main(reference, input):
    # compute the min last metric for demo metrics
    demo_metrics = json.load(open(reference, 'r'))
    demo_min_metrics = collections.defaultdict(lambda:float('inf'))
    for episode_idx, metrics in demo_metrics.items():
        for key, value in metrics.items():
            last_value = value[-1]
            demo_min_metrics[key] = min(demo_min_metrics[key], last_value)
    print(demo_min_metrics)

    # find all metric 
    name = 'metrics_raw.json'
    search_dir = pathlib.Path(input)
    success_rate_map = dict()
    for json_path in search_dir.glob('**/'+name):
        rel_path = json_path.relative_to(search_dir)
        rel_name = str(rel_path.parent)
        this_metrics = json.load(json_path.open('r'))
        metric_success_idxs = collections.defaultdict(list)
        metric_failure_idxs = collections.defaultdict(list)
        for episode_idx, metrics in this_metrics.items():
            for key, value in metrics.items():
                last_value = value[-1]
                # print(episode_idx, key, last_value)
                demo_min = demo_min_metrics[key]
                if last_value >= demo_min:
                    # success
                    metric_success_idxs[key].append(episode_idx)
                else:
                    metric_failure_idxs[key].append(episode_idx)
                # in case of no success
                _ = metric_success_idxs[key]
                _ = metric_failure_idxs[key]
        metric_success_rate = dict()
        n_episodes = len(this_metrics)
        for key, value in metric_success_idxs.items():
            metric_success_rate[key] = len(value) / n_episodes
        # metric_success_rate['failured_idxs'] = metric_failure_idxs
        success_rate_map[rel_name] = metric_success_rate
    
    text = json.dumps(success_rate_map, indent=2)
    print(text)

if __name__ == '__main__':
    main()
