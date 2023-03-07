import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty


def test():
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    example = {
        'cmd': 0,
        'pose': np.zeros((6,))
    }
    queue = SharedMemoryQueue.create_from_examples(
        shm_manager=shm_manager,
        examples=example,
        buffer_size=3
    )
    raised = False
    try:
        queue.get()
    except Empty:
        raised = True
    assert raised

    data = {
        'cmd': 1,
        'pose': np.ones((6,))
    }
    queue.put(data)
    result = queue.get()
    assert result['cmd'] == data['cmd']
    assert np.allclose(result['pose'], data['pose'])

    queue.put(data)
    queue.put(data)
    queue.put(data)
    assert queue.qsize() == 3
    raised = False
    try:
        queue.put(data)
    except Full:
        raised = True
    assert raised

    result = queue.get_all()
    assert np.allclose(result['cmd'], [1,1,1])
    
    queue.put({'cmd': 0})
    queue.put({'cmd': 1})
    queue.put({'cmd': 2})
    queue.get()
    queue.put({'cmd': 3})

    result = queue.get_k(3)
    assert np.allclose(result['cmd'], [1,2,3])

    queue.clear()

if __name__ == "__main__":
    test()
