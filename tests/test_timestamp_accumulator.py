import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import time
from diffusion_policy.common.timestamp_accumulator import (
    get_accumulate_timestamp_idxs,
    TimestampObsAccumulator,
    TimestampActionAccumulator
)


def test_index():
    buffer = np.zeros(16)
    start_time = 0.0
    dt = 1/10

    timestamps = np.linspace(0,1,100)
    gi = list()
    next_global_idx = 0

    local_idxs, global_idxs, next_global_idx = get_accumulate_timestamp_idxs(timestamps, 
        start_time=start_time, dt=dt, next_global_idx=next_global_idx)
    assert local_idxs[0] == 0
    assert global_idxs[0] == 0
    # print(local_idxs)
    # print(global_idxs)
    # print(timestamps[local_idxs])
    buffer[global_idxs] = timestamps[local_idxs]
    gi.extend(global_idxs)
    
    timestamps = np.linspace(0.5,1.5,100)
    local_idxs, global_idxs, next_global_idx = get_accumulate_timestamp_idxs(timestamps, 
        start_time=start_time, dt=dt, next_global_idx = next_global_idx)
    # print(local_idxs)
    # print(global_idxs)
    # print(timestamps[local_idxs])
    # import pdb; pdb.set_trace()
    buffer[global_idxs] = timestamps[local_idxs]
    gi.extend(global_idxs)
    
    assert np.all(buffer[1:] > buffer[:-1])
    assert np.all(np.array(gi) == np.array(list(range(len(gi)))))
    # print(buffer)

    # start over
    next_global_idx = 0
    timestamps = np.linspace(0,1,3)
    local_idxs, global_idxs, next_global_idx = get_accumulate_timestamp_idxs(timestamps, 
        start_time=start_time, dt=dt, next_global_idx = next_global_idx)
    assert local_idxs[0] == 0
    assert local_idxs[-1] == 2
    # print(local_idxs)
    # print(global_idxs)
    # print(timestamps[local_idxs])

    # test numerical error issue
    # this becomes a problem when eps <= 1e-7
    start_time = time.time()
    next_global_idx = 0
    timestamps = np.arange(100000) * dt + start_time
    local_idxs, global_idxs, next_global_idx = get_accumulate_timestamp_idxs(timestamps, 
        start_time=start_time, dt=dt, next_global_idx = next_global_idx)
    assert local_idxs == global_idxs
    # print(local_idxs)
    # print(global_idxs)
    # print(timestamps[local_idxs])


def test_obs_accumulator():
    dt = 1/10
    ddt = 1/100
    n = 100
    d = 6
    start_time = time.time()
    toa = TimestampObsAccumulator(start_time, dt)
    poses = np.arange(n).reshape((n,1))
    poses = np.repeat(poses, d, axis=1)
    timestamps = np.arange(n) * ddt + start_time

    toa.put({
        'pose': poses,
        'timestamp': timestamps
    }, timestamps)
    assert np.all(toa.data['pose'][:,0] == np.arange(10)*10)
    assert len(toa) == 10

    # add the same thing, result shouldn't change
    toa.put({
        'pose': poses,
        'timestamp': timestamps
    }, timestamps)
    assert np.all(toa.data['pose'][:,0] == np.arange(10)*10)
    assert len(toa) == 10
    
    # add lower than desired freuquency to test fill_in
    dt = 1/10
    ddt = 1/5
    n = 10
    d = 6
    start_time = time.time()
    toa = TimestampObsAccumulator(start_time, dt)
    poses = np.arange(n).reshape((n,1))
    poses = np.repeat(poses, d, axis=1)
    timestamps = np.arange(n) * ddt + start_time

    toa.put({
        'pose': poses,
        'timestamp': timestamps
    }, timestamps)
    assert len(toa) == 1 + (n-1) * 2

    timestamps = (np.arange(n) + 2) * ddt + start_time
    toa.put({
        'pose': poses,
        'timestamp': timestamps
    }, timestamps)
    assert len(toa) == 1 + (n-1) * 2 + 4


def test_action_accumulator():
    dt = 1/10
    n = 10
    d = 6
    start_time = time.time()
    taa = TimestampActionAccumulator(start_time, dt)
    actions = np.arange(n).reshape((n,1))
    actions = np.repeat(actions, d, axis=1)

    timestamps = np.arange(n) * dt + start_time
    taa.put(actions, timestamps)
    assert np.all(taa.actions == actions)
    assert np.all(taa.timestamps == timestamps)

    # add another round
    taa.put(actions-5, timestamps-0.5)
    assert np.allclose(taa.timestamps, timestamps)

    # add another round
    taa.put(actions+5, timestamps+0.5)
    assert len(taa) == 15
    assert np.all(taa.actions[:,0] == np.arange(15))
    


if __name__ == '__main__':
    test_action_accumulator()
