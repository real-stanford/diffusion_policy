"""Device selection for the inference entry points (XPU > CUDA > CPU).

An explicit ``--device`` value always wins over detection.
"""
from typing import Optional, Union

import torch


def is_available(device_type: str) -> bool:
    if device_type == 'xpu':
        return hasattr(torch, 'xpu') and torch.xpu.is_available()
    if device_type == 'cuda':
        return torch.cuda.is_available()
    return device_type == 'cpu'


def default_device_type() -> str:
    for device_type in ('xpu', 'cuda'):
        if is_available(device_type):
            return device_type
    return 'cpu'


def resolve_device(device: Union[str, torch.device, None] = None) -> torch.device:
    """None detects the best backend, anything else is honoured verbatim."""
    if device is None:
        device_type = default_device_type()
        return torch.device(device_type, 0) if device_type != 'cpu' else torch.device('cpu')
    return torch.device(device)

