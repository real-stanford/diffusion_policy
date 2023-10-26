import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import time
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer,
    SharedAtomicCounter)


def test():
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    ring_buffer = SharedMemoryRingBuffer.create_from_examples(
        shm_manager,
        {'timestamp': np.array(0, dtype=np.float64)},
        buffer_size=128
    )
    for i in range(30):
        ring_buffer.put({
            'timestamp': np.array(
                time.perf_counter(), 
                dtype=np.float64)
        })
    print(ring_buffer.get())


def _timestamp_worker(ring_buffer, start_event, stop_event):
    while not stop_event.is_set():
        start_event.set()
        ring_buffer.put({
            'timestamp': np.array(
                time.time(), 
                dtype=np.float64)
        })


def test_mp():
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    ring_buffer = SharedMemoryRingBuffer.create_from_examples(
        shm_manager,
        {'timestamp': np.array(0, dtype=np.float64)},
        get_max_k=1,
        get_time_budget=0.01,
        put_desired_frequency=1000
    )
    start_event = mp.Event()
    stop_event = mp.Event()
    worker = mp.Process(target=_timestamp_worker, args=(
        ring_buffer, start_event, stop_event))
    worker.start()
    start_event.wait()
    for i in range(1000):
        t = float(ring_buffer.get()['timestamp'])
        curr_t = time.time()
        print('latency', curr_t - t)
    stop_event.set()
    worker.join()


def test_get_last_k():
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    ring_buffer = SharedMemoryRingBuffer.create_from_examples(
        shm_manager,
        {'counter': np.array(0, dtype=np.int64)},
        buffer_size=8
    )

    from collections import deque
    k = 4
    last_k = deque(maxlen=k)
    for i in range(100):
        ring_buffer.put({
            'counter': np.array(i, dtype=np.int64)
        })
        last_k.append(i)
        if i > k:
            result = ring_buffer.get_last_k(k)['counter']
            assert np.allclose(result, last_k)

    print(ring_buffer.shared_arrays['counter'].get())
    result = ring_buffer.get_last_k(4)
    print(result)


def test_timing():
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    ring_buffer = SharedMemoryRingBuffer.create_from_examples(
        shm_manager,
        {'counter': np.array(0, dtype=np.int64)},
        get_max_k=8,
        get_time_budget=0.1,
        put_desired_frequency=100
    )
    # print(ring_buffer.timestamp_array.get())
    print('buffer_size', ring_buffer.buffer_size)

    dt = 1 / 150
    t_init = time.monotonic()
    for i in range(1000):
        t_start = time.monotonic()
        ring_buffer.put({
            'counter': np.array(i, dtype=np.int64)
        }, wait=False)
        if (i % 10 == 0) and (i > 0):
            result = ring_buffer.get_last_k(8)

        t_end =time.monotonic()
        desired_t = (i+1) * dt + t_init
        if desired_t > t_end:
            time.sleep(desired_t - t_end)
        hz = 1 / (time.monotonic() - t_start)
        print(f'{hz}Hz')


def _timestamp_image_worker(ring_buffer, img_shape, dt, start_event, stop_event):
    i = 0
    t_init = time.monotonic()
    image = np.ones(img_shape, dtype=np.uint8)
    while not stop_event.is_set():
        t_start = time.monotonic()
        start_event.set()
        ring_buffer.put({
            'img': image,
            'timestamp': time.time(),
            'counter': i
        })
        t_end = time.monotonic()
        desired_t = (i+1) * dt + t_init
        # print('alive')
        if desired_t > t_end:
            time.sleep(desired_t - t_end)
        # hz = 1 / (time.monotonic() - t_start)
        i += 1


def test_timing_mp():
    shm_manager = SharedMemoryManager()
    shm_manager.start()

    hz = 200
    img_shape = (1920,1080,3)
    ring_buffer = SharedMemoryRingBuffer.create_from_examples(
        shm_manager,
        examples={
            'img': np.zeros(img_shape, dtype=np.uint8),
            'timestamp': time.time(),
            'counter': 0
        },
        get_max_k=60,
        get_time_budget=0.02,
        put_desired_frequency=hz
    )
    start_event = mp.Event()
    stop_event = mp.Event()
    worker = mp.Process(target=_timestamp_image_worker, args=(
        ring_buffer, img_shape, 1/hz, start_event, stop_event))
    worker.start()
    start_event.wait()
    out = None
    t_start = time.monotonic()
    k = 1
    for i in range(1000):
        if ring_buffer.count < k:
            time.sleep(0)
            continue
        out = ring_buffer.get_last_k(k=k, out=out)
        t = float(out['timestamp'][-1])
        curr_t = time.time()
        print('latency', curr_t - t)
    t_end = time.monotonic()
    print('Get Hz', 1/(t_end-t_start)*1000)
    stop_event.set()
    worker.join()


if __name__ == '__main__':
    # test_mp()
    test_timing_mp()
