from typing import Dict, List, Union

from queue import Empty
import numbers
import time
from multiprocessing.managers import SharedMemoryManager
import numpy as np

from diffusion_policy.shared_memory.shared_ndarray import SharedNDArray
from diffusion_policy.shared_memory.shared_memory_util import ArraySpec, SharedAtomicCounter

class SharedMemoryRingBuffer:
    """
    A Lock-Free FILO Shared Memory Data Structure.
    Stores a sequence of dict of numpy arrays.
    """

    def __init__(self, 
            shm_manager: SharedMemoryManager,
            array_specs: List[ArraySpec],
            get_max_k: int,
            get_time_budget: float,
            put_desired_frequency: float,
            safety_margin: float=1.5
        ):
        """
        shm_manager: Manages the life cycle of share memories 
            across processes. Remember to run .start() before passing.
        array_specs: Name, shape and type of arrays for a single time step.
        get_max_k: The maxmum number of items can be queried at once.
        get_time_budget: The maxmum amount of time spent copying data from 
            shared memory to local memory. Increase this number for larger arrays.
        put_desired_frequency: The maximum frequency that .put() can be called.
            This influces the buffer size.
        """

        # create atomic counter
        counter = SharedAtomicCounter(shm_manager)

        # compute buffer size
        # At any given moment, the past get_max_k items should never 
        # be touched (to be read freely). Assuming the reading is reading
        # these k items, which takes maximum of get_time_budget seconds,
        # we need enough empty slots to make sure put_desired_frequency Hz
        # of put can be sustaied.
        buffer_size = int(np.ceil(
            put_desired_frequency * get_time_budget 
            * safety_margin)) + get_max_k

        # allocate shared memory
        shared_arrays = dict()
        for spec in array_specs:
            key = spec.name
            assert key not in shared_arrays
            array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(buffer_size,) + tuple(spec.shape),
                dtype=spec.dtype)
            shared_arrays[key] = array
        
        # allocate timestamp array
        timestamp_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager, 
            shape=(buffer_size,),
            dtype=np.float64)
        timestamp_array.get()[:] = -np.inf
        
        self.buffer_size = buffer_size
        self.array_specs = array_specs
        self.counter = counter
        self.shared_arrays = shared_arrays
        self.timestamp_array = timestamp_array
        self.get_time_budget = get_time_budget
        self.get_max_k = get_max_k
        self.put_desired_frequency = put_desired_frequency

    
    @property
    def count(self):
        return self.counter.load()
    
    @classmethod
    def create_from_examples(cls, 
            shm_manager: SharedMemoryManager,
            examples: Dict[str, Union[np.ndarray, numbers.Number]], 
            get_max_k: int=32,
            get_time_budget: float=0.01,
            put_desired_frequency: float=60
            ):
        specs = list()
        for key, value in examples.items():
            shape = None
            dtype = None
            if isinstance(value, np.ndarray):
                shape = value.shape
                dtype = value.dtype
                assert dtype != np.dtype('O')
            elif isinstance(value, numbers.Number):
                shape = tuple()
                dtype = np.dtype(type(value))
            else:
                raise TypeError(f'Unsupported type {type(value)}')

            spec = ArraySpec(
                name=key,
                shape=shape,
                dtype=dtype
            )
            specs.append(spec)

        obj = cls(
            shm_manager=shm_manager,
            array_specs=specs,
            get_max_k=get_max_k,
            get_time_budget=get_time_budget,
            put_desired_frequency=put_desired_frequency
            )
        return obj

    def clear(self):
        self.counter.store(0)
    
    def put(self, data: Dict[str, Union[np.ndarray, numbers.Number]], wait: bool=True):
        count = self.counter.load()
        next_idx = count % self.buffer_size
        # Make sure the next self.get_max_k elements in the ring buffer have at least 
        # self.get_time_budget seconds untouched after written, so that
        # get_last_k can safely read k elements from any count location.
        # Sanity check: when get_max_k == 1, the element pointed by next_idx
        # should be rewritten at minimum self.get_time_budget seconds later.
        timestamp_lookahead_idx = (next_idx + self.get_max_k - 1) % self.buffer_size
        old_timestamp = self.timestamp_array.get()[timestamp_lookahead_idx]
        t = time.monotonic()
        if (t - old_timestamp) < self.get_time_budget:
            deltat = t - old_timestamp
            if wait:
                # sleep the remaining time to be safe
                time.sleep(self.get_time_budget - deltat)
            else:
                # throw an error
                past_iters = self.buffer_size - self.get_max_k
                hz = past_iters / deltat
                raise TimeoutError(
                    'Put executed too fast {}items/{:.4f}s ~= {}Hz'.format(
                        past_iters, deltat,hz))

        # write to shared memory
        for key, value in data.items():
            arr: np.ndarray
            arr = self.shared_arrays[key].get()
            if isinstance(value, np.ndarray):
                arr[next_idx] = value
            else:
                arr[next_idx] = np.array(value, dtype=arr.dtype)
        
        # update timestamp
        self.timestamp_array.get()[next_idx] = time.monotonic()
        self.counter.add(1)

    def _allocate_empty(self, k=None):
        result = dict()
        for spec in self.array_specs:
            shape = spec.shape
            if k is not None:
                shape = (k,) + shape
            result[spec.name] = np.empty(
                shape=shape, dtype=spec.dtype)
        return result

    def get(self, out=None) -> Dict[str, np.ndarray]:
        if out is None:
            out = self._allocate_empty()
        start_time = time.monotonic()
        count = self.counter.load()
        curr_idx = (count - 1) % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            np.copyto(out[key], arr[curr_idx])
        end_time = time.monotonic()
        dt = end_time - start_time
        if dt > self.get_time_budget:
            raise TimeoutError(f'Get time out {dt} vs {self.get_time_budget}')
        return out
    
    def get_last_k(self, k:int, out=None) -> Dict[str, np.ndarray]:
        assert k <= self.get_max_k
        if out is None:
            out = self._allocate_empty(k)
        start_time = time.monotonic()
        count = self.counter.load()
        assert k <= count
        curr_idx = (count - 1) % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            target = out[key]

            end = curr_idx + 1
            start = max(0, end - k)
            target_end = k
            target_start = target_end - (end - start)
            target[target_start: target_end] = arr[start:end]

            remainder = k - (end - start)
            if remainder > 0:
                # wrap around
                end = self.buffer_size
                start = end - remainder
                target_start = 0
                target_end = end - start
                target[target_start: target_end] = arr[start:end]
        end_time = time.monotonic()
        dt = end_time - start_time
        if dt > self.get_time_budget:
            raise TimeoutError(f'Get time out {dt} vs {self.get_time_budget}')
        return out

    def get_all(self) -> Dict[str, np.ndarray]:
        k = min(self.count, self.get_max_k)
        return self.get_last_k(k=k)
