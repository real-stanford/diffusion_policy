from typing import List, Tuple, Optional, Dict
import math
import numpy as np


def get_accumulate_timestamp_idxs(
    timestamps: List[float],  
    start_time: float, 
    dt: float, 
    eps:float=1e-5,
    next_global_idx: Optional[int]=0,
    allow_negative=False
    ) -> Tuple[List[int], List[int], int]:
    """
    For each dt window, choose the first timestamp in the window.
    Assumes timestamps sorted. One timestamp might be chosen multiple times due to dropped frames.
    next_global_idx should start at 0 normally, and then use the returned next_global_idx. 
    However, when overwiting previous values are desired, set last_global_idx to None.

    Returns:
    local_idxs: which index in the given timestamps array to chose from
    global_idxs: the global index of each chosen timestamp
    next_global_idx: used for next call.
    """
    local_idxs = list()
    global_idxs = list()
    for local_idx, ts in enumerate(timestamps):
        # add eps * dt to timestamps so that when ts == start_time + k * dt 
        # is always recorded as kth element (avoiding floating point errors)
        global_idx = math.floor((ts - start_time) / dt + eps)
        if (not allow_negative) and (global_idx < 0):
            continue
        if next_global_idx is None:
            next_global_idx = global_idx

        n_repeats = max(0, global_idx - next_global_idx + 1)
        for i in range(n_repeats):
            local_idxs.append(local_idx)
            global_idxs.append(next_global_idx + i)
        next_global_idx += n_repeats
    return local_idxs, global_idxs, next_global_idx


def align_timestamps(    
        timestamps: List[float], 
        target_global_idxs: List[int], 
        start_time: float, 
        dt: float, 
        eps:float=1e-5):
    if isinstance(target_global_idxs, np.ndarray):
        target_global_idxs = target_global_idxs.tolist()
    assert len(target_global_idxs) > 0

    local_idxs, global_idxs, _ = get_accumulate_timestamp_idxs(
        timestamps=timestamps,
        start_time=start_time,
        dt=dt,
        eps=eps,
        next_global_idx=target_global_idxs[0],
        allow_negative=True
    )
    if len(global_idxs) > len(target_global_idxs):
        # if more steps available, truncate
        global_idxs = global_idxs[:len(target_global_idxs)]
        local_idxs = local_idxs[:len(target_global_idxs)]
    
    if len(global_idxs) == 0:
        import pdb; pdb.set_trace()

    for i in range(len(target_global_idxs) - len(global_idxs)):
        # if missing, repeat
        local_idxs.append(len(timestamps)-1)
        global_idxs.append(global_idxs[-1] + 1)
    assert global_idxs == target_global_idxs
    assert len(local_idxs) == len(global_idxs)
    return local_idxs


class TimestampObsAccumulator:
    def __init__(self, 
            start_time: float, 
            dt: float, 
            eps: float=1e-5):
        self.start_time = start_time
        self.dt = dt
        self.eps = eps
        self.obs_buffer = dict()
        self.timestamp_buffer = None
        self.next_global_idx = 0
    
    def __len__(self):
        return self.next_global_idx
    
    @property
    def data(self):
        if self.timestamp_buffer is None:
            return dict()
        result = dict()
        for key, value in self.obs_buffer.items():
            result[key] = value[:len(self)]
        return result

    @property
    def actual_timestamps(self):
        if self.timestamp_buffer is None:
            return np.array([])
        return self.timestamp_buffer[:len(self)]
    
    @property
    def timestamps(self):
        if self.timestamp_buffer is None:
            return np.array([])
        return self.start_time + np.arange(len(self)) * self.dt

    def put(self, data: Dict[str, np.ndarray], timestamps: np.ndarray):
        """
        data:
            key: T,*
        """

        local_idxs, global_idxs, self.next_global_idx = get_accumulate_timestamp_idxs(
            timestamps=timestamps,
            start_time=self.start_time,
            dt=self.dt,
            eps=self.eps,
            next_global_idx=self.next_global_idx
        )

        if len(global_idxs) > 0:
            if self.timestamp_buffer is None:
                # first allocation
                self.obs_buffer = dict()
                for key, value in data.items():
                    self.obs_buffer[key] = np.zeros_like(value)
                self.timestamp_buffer = np.zeros(
                    (len(timestamps),), dtype=np.float64)
            
            this_max_size = global_idxs[-1] + 1
            if this_max_size > len(self.timestamp_buffer):
                # reallocate
                new_size = max(this_max_size, len(self.timestamp_buffer) * 2)
                for key in list(self.obs_buffer.keys()):
                    new_shape = (new_size,) + self.obs_buffer[key].shape[1:]
                    self.obs_buffer[key] = np.resize(self.obs_buffer[key], new_shape)
                self.timestamp_buffer = np.resize(self.timestamp_buffer, (new_size))
            
            # write data
            for key, value in self.obs_buffer.items():
                value[global_idxs] = data[key][local_idxs]
            self.timestamp_buffer[global_idxs] = timestamps[local_idxs]


class TimestampActionAccumulator:
    def __init__(self, 
            start_time: float, 
            dt: float, 
            eps: float=1e-5):
        """
        Different from Obs accumulator, the action accumulator
        allows overwriting previous values.
        """
        self.start_time = start_time
        self.dt = dt
        self.eps = eps
        self.action_buffer = None
        self.timestamp_buffer = None
        self.size = 0
    
    def __len__(self):
        return self.size
    
    @property
    def actions(self):
        if self.action_buffer is None:
            return np.array([])
        return self.action_buffer[:len(self)]
    
    @property
    def actual_timestamps(self):
        if self.timestamp_buffer is None:
            return np.array([])
        return self.timestamp_buffer[:len(self)]
    
    @property
    def timestamps(self):
        if self.timestamp_buffer is None:
            return np.array([])
        return self.start_time + np.arange(len(self)) * self.dt

    def put(self, actions: np.ndarray, timestamps: np.ndarray):
        """
        Note: timestamps is the time when the action will be issued, 
        not when the action will be completed (target_timestamp)
        """

        local_idxs, global_idxs, _ = get_accumulate_timestamp_idxs(
            timestamps=timestamps,
            start_time=self.start_time,
            dt=self.dt,
            eps=self.eps,
            # allows overwriting previous actions
            next_global_idx=None
        )

        if len(global_idxs) > 0:
            if self.timestamp_buffer is None:
                # first allocation
                self.action_buffer = np.zeros_like(actions)
                self.timestamp_buffer = np.zeros((len(actions),), dtype=np.float64)

            this_max_size = global_idxs[-1] + 1
            if this_max_size > len(self.timestamp_buffer):
                # reallocate
                new_size = max(this_max_size, len(self.timestamp_buffer) * 2)
                new_shape = (new_size,) + self.action_buffer.shape[1:]
                self.action_buffer = np.resize(self.action_buffer, new_shape)
                self.timestamp_buffer = np.resize(self.timestamp_buffer, (new_size,))
            
            # potentially rewrite old data (as expected)
            self.action_buffer[global_idxs] = actions[local_idxs]
            self.timestamp_buffer[global_idxs] = timestamps[local_idxs]
            self.size = max(self.size, this_max_size)
