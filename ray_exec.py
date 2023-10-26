"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace -- logger.mode=online
"""
import os
import ray
import click

def worker_fn(command_args, data_src=None, unbuffer_python=False, use_shell=False):
    import os
    import subprocess
    import signal
    import time

    # setup data symlink
    if data_src is not None:
        cwd = os.getcwd()
        src = os.path.expanduser(data_src)
        dst = os.path.join(cwd, 'data')
        try:
            os.symlink(src=src, dst=dst)
        except FileExistsError:
            # it's fine if it already exists
            pass

    # run command
    process_env = os.environ.copy()
    if unbuffer_python:
        # disable stdout/stderr buffering for subprocess (if python)
        # to remove latency between print statement and receiving printed result
        process_env['PYTHONUNBUFFERED'] = 'TRUE'
    
    # ray worker masks out Ctrl-C signal (ie SIGINT)
    # here we unblock this signal for the child process
    def preexec_function():
        import signal
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
    
    if use_shell:
        command_args = ' '.join(command_args)

    # stdout passthrough to ray worker, which is then passed to ray driver
    process = subprocess.Popen(
        args=command_args, 
        env=process_env,
        preexec_fn=preexec_function,
        shell=use_shell)

    while process.poll() is None:
        try:
            # sleep to ensure that monitor thread can acquire gil
            # and raise KeyboardInterrupt here. 
            time.sleep(0.01)
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            print('SIGINT sent to subprocess')
        except Exception as e:
            process.terminate()
            raise e

    if process.returncode not in (0, -2):
        print("Failed execution!")
        raise RuntimeError("Failed execution.")
    return process.returncode


@click.command()
@click.option('--ray_address', '-ra', default='auto')
@click.option('--num_cpus', '-nc', default=7, type=float)
@click.option('--num_gpus', '-ng', default=1, type=float)
@click.option('--max_retries', '-mr', default=0, type=int)
@click.option('--data_src', '-d', default='./data', type=str)
@click.option('--unbuffer_python', '-u', is_flag=True, default=False)
@click.argument('command_args', nargs=-1, type=str)
def main(ray_address, 
    num_cpus, num_gpus, max_retries, 
    data_src, unbuffer_python, 
    command_args):
    # expand path
    if data_src is not None:
        data_src = os.path.abspath(os.path.expanduser(data_src))

    # init ray
    root_dir = os.path.dirname(__file__)
    runtime_env = {
        'working_dir': root_dir,
        'excludes': ['.git']
    }
    ray.init(
        address=ray_address, 
        runtime_env=runtime_env
    )
    # remote worker func
    worker_ray = ray.remote(worker_fn).options(
        num_cpus=num_cpus, 
        num_gpus=num_gpus,
        max_retries=max_retries,
        # resources=resources,
        retry_exceptions=True
        )
    # run
    task_ref = worker_ray.remote(command_args, data_src, unbuffer_python)

    try:
        # normal case
        result = ray.get(task_ref)
        print('Return code: ', result)
    except KeyboardInterrupt:
        # a KeyboardInterrupt will be raised in worker
        ray.cancel(task_ref, force=False)
        result = ray.get(task_ref)
        print('Return code: ', result)
    except Exception as e:
        # worker will be terminated
        ray.cancel(task_ref, force=True)
        raise e
    

if __name__ == '__main__':
    main()
