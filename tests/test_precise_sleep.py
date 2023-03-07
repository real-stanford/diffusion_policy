import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import time
import numpy as np
from diffusion_policy.common.precise_sleep import precise_sleep, precise_wait


def test_sleep():
    dt = 0.1
    tol = 1e-3
    time_samples = list()
    for i in range(100):
        precise_sleep(dt)
        # time.sleep(dt)
        time_samples.append(time.monotonic())
    time_deltas = np.diff(time_samples)

    from matplotlib import pyplot as plt
    plt.plot(time_deltas)
    plt.ylim((dt-tol,dt+tol))


def test_wait():
    dt = 0.1
    tol = 1e-3
    errors = list()
    t_start = time.monotonic()
    for i in range(1,100):
        t_end_desired = t_start + i * dt
        time.sleep(t_end_desired - time.monotonic())
        t_end = time.monotonic()
        errors.append(t_end - t_end_desired)

    new_errors = list()
    t_start = time.monotonic()
    for i in range(1,100):
        t_end_desired = t_start + i * dt
        precise_wait(t_end_desired)
        t_end = time.monotonic()
        new_errors.append(t_end - t_end_desired)
    
    from matplotlib import pyplot as plt
    plt.plot(errors, label='time.sleep')
    plt.plot(new_errors, label='sleep/spin hybrid')
    plt.ylim((-tol,+tol))
    plt.title('0.1 sec sleep error')
    plt.legend()

    
if __name__ == '__main__':
    test_sleep()
