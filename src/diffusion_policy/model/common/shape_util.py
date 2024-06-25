from typing import Dict, List, Tuple, Callable
import torch
import torch.nn as nn

def get_module_device(m: nn.Module):
    device = torch.device('cpu')
    try:
        param = next(iter(m.parameters()))
        device = param.device
    except StopIteration:
        pass
    return device

@torch.no_grad()
def get_output_shape(
        input_shape: Tuple[int],
        net: Callable[[torch.Tensor], torch.Tensor]
    ):  
        device = get_module_device(net)
        test_input = torch.zeros((1,)+tuple(input_shape), device=device)
        test_output = net(test_input)
        output_shape = tuple(test_output.shape[1:])
        return output_shape
