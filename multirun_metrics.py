from typing import List, Optional
import pathlib
import pandas as pd
import numpy as np
import numba
import click
import time
import collections
import json
import wandb
import yaml
import numbers
import scipy.ndimage as sn
from diffusion_policy.common.json_logger import read_json_log, JsonLogger
import logging

@numba.jit(nopython=True)
def get_indexed_window_average(
        arr: np.ndarray, idxs: np.ndarray, window_size: int):
    result = np.zeros(idxs.shape, dtype=arr.dtype)
    length = arr.shape[0]
    for i in range(len(idxs)):
        idx = idxs[i]
        start = max(idx - window_size, 0)
        end = min(start + window_size, length)
        result[i] = np.mean(arr[start:end])
    return result


def compute_metrics(log_df: pd.DataFrame, key: str, 
        end_step: Optional[int]=None,
        k_min_loss: int=10,
        k_around_max: int=10,
        max_k_window: int=10,
        replace_slash: int=True,
        ):
    if key not in log_df:
        return dict()

    # prepare data
    if end_step is not None:
        log_df = log_df.iloc[:end_step]
    is_key = ~pd.isnull(log_df[key])
    is_key_idxs = is_key.index[is_key].to_numpy()
    if len(is_key_idxs) == 0:
        return dict()

    key_data = log_df[key][is_key].to_numpy()
    # after adding validation to workspace
    # rollout happens at the last step of each epoch
    # where the reported train_loss and val_loss
    # are already the average for that epoch
    train_loss = log_df['train_loss'][is_key].to_numpy()
    val_loss = log_df['val_loss'][is_key].to_numpy()

    result = dict()

    log_key = key
    if replace_slash:
        log_key = key.replace('/', '_')
    # max
    max_value = np.max(key_data)
    result['max/'+log_key] = max_value

    # k_around_max
    max_idx = np.argmax(key_data)
    end = min(max_idx + k_around_max // 2, len(key_data))
    start = max(end - k_around_max, 0)
    k_around_max_value = np.mean(key_data[start:end])
    result['k_around_max/'+log_key] = k_around_max_value

    # max_k_window
    k_window_value = sn.uniform_filter1d(key_data, size=max_k_window, axis=0, mode='nearest')
    max_k_window_value = np.max(k_window_value)
    result['max_k_window/'+log_key] = max_k_window_value

    # min_train_loss
    min_idx = np.argmin(train_loss)
    min_train_loss_value = key_data[min_idx]
    result['min_train_loss/'+log_key] = min_train_loss_value

    # min_val_loss
    min_idx = np.argmin(val_loss)
    min_val_loss_value = key_data[min_idx]
    result['min_val_loss/'+log_key] = min_val_loss_value

    # k_min_train_loss
    min_loss_idxs = np.argsort(train_loss)[:k_min_loss]
    k_min_train_loss_value = np.mean(key_data[min_loss_idxs])
    result['k_min_train_loss/'+log_key] = k_min_train_loss_value

    # k_min_val_loss
    min_loss_idxs = np.argsort(val_loss)[:k_min_loss]
    k_min_val_loss_value = np.mean(key_data[min_loss_idxs])
    result['k_min_val_loss/'+log_key] = k_min_val_loss_value

    # last
    result['last/'+log_key] = key_data[-1]

    # global step for visualization
    result['metric_global_step/'+log_key] = is_key_idxs[-1]
    return result


def compute_metrics_agg(
        log_dfs: List[pd.DataFrame], 
        key: str, end_step:int, 
        **kwargs):
    
    # compute metrics
    results = collections.defaultdict(list)
    for log_df in log_dfs:
        result = compute_metrics(log_df, key=key, end_step=end_step, **kwargs)
        for k, v in result.items():
            results[k].append(v)
    # agg
    agg_result = dict()
    for k, v in results.items():
        value = np.mean(v)
        if k.startswith('metric_global_step'):
            value = int(value)
        agg_result[k] = value
    return agg_result


@click.command()
@click.option('--input', '-i', required=True, help='Root logging dir, contains train_* dirs')
@click.option('--key', '-k', multiple=True, default=['test/mean_score'])
@click.option('--interval', default=10, type=float)
@click.option('--replace_slash', default=True, type=bool)
@click.option('--index_key', '-ik', default='global_step')
@click.option('--use_wandb', '-w', is_flag=True, default=False)
@click.option('--project', default=None)
@click.option('--name', default=None)
@click.option('--id', default=None)
@click.option('--group', default=None)
def main(
    input,
    key,
    interval,
    replace_slash,
    index_key,
    use_wandb,
    # wandb args
    project,
    name,
    id,
    group):
    root_dir = pathlib.Path(input)
    assert root_dir.is_dir()
    metrics_dir = root_dir.joinpath('metrics')
    metrics_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(metrics_dir.joinpath("metrics.log"))),
            logging.StreamHandler()
        ]
    )
    
    train_dirs = list(root_dir.glob('train_*'))
    log_files = [x.joinpath('logs.json.txt') for x in train_dirs]
    logging.info("Monitor waiting for log files!")
    while True:
        # wait for files to show up
        files_exist = True
        for log_file in log_files:
            if not log_file.is_file():
                files_exist = False
        if files_exist:
            break
        time.sleep(1.0)
    logging.info("All log files ready!")

    # init path
    metric_log_path = metrics_dir.joinpath('logs.json.txt')
    metric_path = metrics_dir.joinpath('metrics.json')
    config_path = root_dir.joinpath('config.yaml')

    # load config
    config = yaml.safe_load(config_path.open('r'))

    # init wandb
    wandb_run = None
    if use_wandb:
        wandb_kwargs = config['logging']
        if project is not None:
            wandb_kwargs['project'] = project
        if id is not None:
            wandb_kwargs['id'] = id
        if name is not None:
            wandb_kwargs['name'] = name
        if group is not None:
            wandb_kwargs['group'] = group
        wandb_kwargs['resume'] = True
        wandb_run = wandb.init(
            dir=str(metrics_dir),
            config=config,
            # auto-resume run, automatically load id
            # as long as using the same dir.
            # https://docs.wandb.ai/guides/track/advanced/resuming#resuming-guidance
            **wandb_kwargs
        )
        wandb.config.update(
            {
                "output_dir": str(root_dir),
            }
        )

    with JsonLogger(metric_log_path) as json_logger:
        last_log = json_logger.get_last_log()
        while True:
            # read json files
            log_dfs = [read_json_log(str(x), required_keys=key) for x in log_files]

            # previously logged data point
            last_log_idx = -1
            if last_log is not None:
                last_log_idx = log_dfs[0].index[log_dfs[0][index_key] <= last_log[index_key]][-1]
            
            start_idx = last_log_idx + 1
            # last idx where we have a data point from all logs
            end_idx = min(*[len(x) for x in log_dfs])

            # log every position
            for this_idx in range(start_idx, end_idx):
                # compute metrics
                all_metrics = dict()
                global_step = log_dfs[0]['global_step'][this_idx]
                epoch = log_dfs[0]['epoch'][this_idx]
                all_metrics['global_step'] = global_step
                all_metrics['epoch'] = epoch
                for k in key:
                    metrics = compute_metrics_agg(
                        log_dfs=log_dfs, key=k, end_step=this_idx+1,
                        replace_slash=replace_slash)
                    all_metrics.update(metrics)

                # sanitize metrics
                old_metrics = all_metrics
                all_metrics = dict()
                for k, v in old_metrics.items():
                    if isinstance(v, numbers.Integral):
                        all_metrics[k] = int(v)
                    elif isinstance(v, numbers.Number):
                        all_metrics[k] = float(v)
                
                has_update = all_metrics != last_log
                if has_update:
                    last_log = all_metrics
                    json_logger.log(all_metrics)

                    with metric_path.open('w') as f:
                        json.dump(all_metrics, f, sort_keys=True, indent=2)

                    if wandb_run is not None:
                        wandb_run.log(all_metrics, step=all_metrics[index_key])

                    logging.info(f"Metrics logged at step {all_metrics[index_key]}")
            
            time.sleep(interval)


if __name__ == "__main__":
    main()
