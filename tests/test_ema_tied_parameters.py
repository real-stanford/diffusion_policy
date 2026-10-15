import copy
import pytest
import torch
from diffusion_policy.model.diffusion.ema_model import EMAModel

class TiedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Linear(2, 2, bias=False)
        self.output = torch.nn.Linear(2, 2, bias=False)
        self.output.weight = self.input.weight
        self.other = torch.nn.Linear(2, 2, bias=False)

@pytest.mark.parametrize("step", [2, 3, 8])
def test_shared_parameter_receives_one_ema_update(step):
    model = TiedModel()
    ema = EMAModel(copy.deepcopy(model), power=1., inv_gamma=1.)
    with torch.no_grad():
        model.input.weight.fill_(10.)
        model.other.weight.fill_(6.)
        ema.averaged_model.input.weight.fill_(0.)
        ema.averaged_model.other.weight.fill_(2.)
    ema.optimization_step = step
    decay = ema.get_decay(step)
    ema.step(model)
    torch.testing.assert_close(ema.averaged_model.input.weight,
                              torch.full((2, 2), 10. * (1 - decay)))
    torch.testing.assert_close(ema.averaged_model.other.weight,
                              torch.full((2, 2), 2. * decay + 6. * (1 - decay)))
    assert ema.averaged_model.input.weight is ema.averaged_model.output.weight
    assert model.input.weight is model.output.weight


def test_repeated_tied_updates_match_scalar_recurrence():
    model = TiedModel()
    ema = EMAModel(copy.deepcopy(model), power=.75)
    expected = ema.averaged_model.input.weight.clone()
    for value in (2., 4., 1., -3., 7., 10.):
        with torch.no_grad():
            model.input.weight.fill_(value)
        decay = ema.get_decay(ema.optimization_step)
        expected = expected * decay + value * (1 - decay)
        ema.step(model)
        torch.testing.assert_close(ema.averaged_model.input.weight, expected)
        assert ema.averaged_model.input.weight is ema.averaged_model.output.weight
