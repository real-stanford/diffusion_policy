from typing import Tuple
from dataclasses import dataclass
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from atomics import atomicview, MemoryOrder, UINT

@dataclass
class ArraySpec:
    name: str
    shape: Tuple[int]
    dtype: np.dtype


class SharedAtomicCounter:
    def __init__(self, 
            shm_manager: SharedMemoryManager, 
            size :int=8 # 64bit int
            ):
        shm = shm_manager.SharedMemory(size=size)
        self.shm = shm
        self.size = size
        self.store(0) # initialize

    @property
    def buf(self):
        return self.shm.buf[:self.size]

    def load(self) -> int:
        with atomicview(buffer=self.buf, atype=UINT) as a: 
            value = a.load(order=MemoryOrder.ACQUIRE)
        return value
    
    def store(self, value: int):
        with atomicview(buffer=self.buf, atype=UINT) as a:
            a.store(value, order=MemoryOrder.RELEASE)
    
    def add(self, value: int):
        with atomicview(buffer=self.buf, atype=UINT) as a:
            a.add(value, order=MemoryOrder.ACQ_REL)
