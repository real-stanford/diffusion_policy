"""
Start local ray cluster
(robodiff)$ export CUDA_VISIBLE_DEVICES=0,1,2 # select GPUs to be managed by the ray cluster
(robodiff)$ ray start --head --num-gpus=3

Training:
python ray_train_multirun.py --config-name=train_diffusion_unet_lowdim_workspace --seeds=42,43,44 --monitor_key=test/mean_score -- logger.mode=online training.eval_first=True
"""
import os
import ray
import click
import hydra
import yaml
import wandb
import pathlib
import collections
from pprint import pprint
from omegaconf import OmegaConf
from ray_exec import worker_fn
from ray.util.placement_group import (
    placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--config-name', '-cn', required=True, type=str)
@click.option('--config-dir', '-cd', default=None, type=str)
@click.option('--seeds', '-s', default='42,43,44', type=str)
@click.option('--monitor_key', '-k', multiple=True, default=['test/mean_score'])
@click.option('--ray_address', '-ra', default='auto')
@click.option('--num_cpus', '-nc', default=7, type=float)
@click.option('--num_gpus', '-ng', default=1, type=float)
@click.option('--max_retries', '-mr', default=0, type=int)
@click.option('--monitor_max_retires', default=3, type=int)
@click.option('--data_src', '-d', default='./data', type=str)
@click.option('--unbuffer_python', '-u', is_flag=True, default=False)
@click.option('--single_node', '-sn', is_flag=True, default=False, help='run all experiments on a single machine')
@click.argument('command_args', nargs=-1, type=str)
def main(config_name, config_dir, seeds, monitor_key, ray_address, 
    num_cpus, num_gpus, max_retries, monitor_max_retires,
    data_src, unbuffer_python, 
    single_node, command_args):
    # parse args
    seeds = [int(x) for x in seeds.split(',')]
    # expand path
    if data_src is not None:
        data_src = os.path.abspath(os.path.expanduser(data_src))

    # initialize hydra
    if config_dir is None:
        config_path_abs = pathlib.Path(__file__).parent.joinpath(
            'diffusion_policy','config')
        config_path_rel = str(config_path_abs.relative_to(pathlib.Path.cwd()))
    else:
        config_path_rel = config_dir

    run_command_args = list()
    monitor_command_args = list()
    with hydra.initialize(
        version_base=None, 
        config_path=config_path_rel):

        # generate raw config
        cfg = hydra.compose(
            config_name=config_name, 
            overrides=command_args)
        OmegaConf.resolve(cfg)
    
        # manually create output dir
        output_dir = pathlib.Path(cfg.multi_run.run_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        config_path = output_dir.joinpath('config.yaml')
        print(output_dir)

        # save current config
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), 
            config_path.open('w'), default_flow_style=False)

        # wandb
        wandb_group_id = wandb.util.generate_id()
        name_base = cfg.multi_run.wandb_name_base

        # create monitor command args
        monitor_command_args = [
            'python',
            'multirun_metrics.py',
            '--input', str(output_dir),
            '--use_wandb',
            '--project', 'diffusion_policy_metrics',
            '--group', wandb_group_id
        ]
        for k in monitor_key:
            monitor_command_args.extend([
                '--key', k
            ])

        # generate command args
        run_command_args = list()
        for i, seed in enumerate(seeds):
            test_start_seed = (seed + 1) * 100000
            this_output_dir = output_dir.joinpath(f'train_{i}')
            this_output_dir.mkdir()
            wandb_name = name_base + f'_train_{i}'
            wandb_run_id = wandb_group_id + f'_train_{i}'

            this_command_args = [
                'python',
                'train.py',
                '--config-name='+config_name,
                '--config-dir='+config_path_rel
            ]

            this_command_args.extend(command_args)
            this_command_args.extend([
                f'training.seed={seed}',
                f'task.env_runner.test_start_seed={test_start_seed}',
                f'logging.name={wandb_name}',
                f'logging.id={wandb_run_id}',
                f'logging.group={wandb_group_id}',
                f'hydra.run.dir={this_output_dir}'
            ])
            run_command_args.append(this_command_args)

    # init ray
    root_dir = os.path.dirname(__file__)
    runtime_env = {
        'working_dir': root_dir,
        'excludes': ['.git'],
        'pip': ['dm-control==1.0.9']
    }
    ray.init(
        address=ray_address, 
        runtime_env=runtime_env
    )
    
    # create resources for train
    train_resources = dict()

    train_bundle = dict(train_resources)
    train_bundle['CPU'] = num_cpus
    train_bundle['GPU'] = num_gpus

    # create resources for monitor
    monitor_resources = dict()
    monitor_resources['CPU'] = 1
    
    monitor_bundle = dict(monitor_resources)

    # aggregate bundle
    bundle = collections.defaultdict(lambda:0)
    n_train_bundles = 1
    if single_node:
        n_train_bundles = len(seeds)
    for _ in range(n_train_bundles):
        for k, v in train_bundle.items():
            bundle[k] += v
    for k, v in monitor_bundle.items():
        bundle[k] += v
    bundle = dict(bundle)

    # create placement group
    print("Creating placement group with resources:")
    pprint(bundle)
    pg = placement_group([bundle])

    # run
    task_name_map = dict()
    task_refs = list()
    for i, this_command_args in enumerate(run_command_args):
        if single_node or i == (len(run_command_args) - 1):
            print(f'Training worker {i} with placement group.')
            ray.get(pg.ready())
            print("Placement Group created!")
            worker_ray = ray.remote(worker_fn).options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                max_retries=max_retries,
                resources=train_resources,
                retry_exceptions=True,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg)
            )
        else:
            print(f'Training worker {i} without placement group.')
            worker_ray = ray.remote(worker_fn).options(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                max_retries=max_retries,
                resources=train_resources,
                retry_exceptions=True,
            )
        task_ref = worker_ray.remote(
            this_command_args, data_src, unbuffer_python)
        task_refs.append(task_ref)
        task_name_map[task_ref] = f'train_{i}'

    # monitor worker is always packed on the same node
    # as training worker 0
    ray.get(pg.ready())
    monitor_worker_ray = ray.remote(worker_fn).options(
        num_cpus=1, 
        num_gpus=0,
        max_retries=monitor_max_retires,
        # resources=monitor_resources,
        retry_exceptions=True,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg)
        )
    monitor_ref = monitor_worker_ray.remote(
            monitor_command_args, data_src, unbuffer_python)
    task_name_map[monitor_ref] = 'metrics'

    try:
        # normal case
        ready_refs = list()
        rest_refs = task_refs
        while len(ready_refs) < len(task_refs):
            this_ready_refs, rest_refs = ray.wait(rest_refs, 
                num_returns=1, timeout=None, fetch_local=True)
            cancel_other_tasks = False
            for ref in this_ready_refs:
                task_name = task_name_map[ref]
                try:
                    result = ray.get(ref)
                    print(f"Task {task_name} finished with result: {result}")
                except KeyboardInterrupt as e:
                    # skip to outer try catch
                    raise KeyboardInterrupt
                except Exception as e:
                    print(f"Task {task_name} raised exception: {e}")
                    this_cancel_other_tasks = True
                    if isinstance(e, ray.exceptions.RayTaskError):
                        if isinstance(e.cause, ray.exceptions.TaskCancelledError):
                            this_cancel_other_tasks = False
                    cancel_other_tasks = cancel_other_tasks or this_cancel_other_tasks
                ready_refs.append(ref)
            if cancel_other_tasks:
                print('Exception! Cancelling all other tasks.')
                # cancel all other refs
                for _ref in rest_refs:
                    ray.cancel(_ref, force=False)
        print("Training tasks done.")
        ray.cancel(monitor_ref, force=False)
    except KeyboardInterrupt:
        print('KeyboardInterrupt received in the driver.')
        # a KeyboardInterrupt will be raised in worker
        _ = [ray.cancel(x, force=False) for x in task_refs + [monitor_ref]]
        print('KeyboardInterrupt sent to workers.')
    except Exception as e:
        # worker will be terminated
        _ = [ray.cancel(x, force=True) for x in task_refs + [monitor_ref]]
        raise e

    for ref in task_refs + [monitor_ref]:
        task_name = task_name_map[ref]
        try:
            result = ray.get(ref)
            print(f"Task {task_name} finished with result: {result}")
        except KeyboardInterrupt as e:
            # force kill everything.
            print("Force killing all workers")
            _ = [ray.cancel(x, force=True) for x in task_refs]
            ray.cancel(monitor_ref, force=True)
        except Exception as e:
            print(f"Task {task_name} raised exception: {e}")


if __name__ == "__main__":
    main()
