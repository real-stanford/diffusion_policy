import copy
import torch
from diffusion_policy.model.diffusion.ema_model import EMAModel


def test_ema_copies_batchnorm_statistics_for_inference():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
    ema = EMAModel(copy.deepcopy(model))
    model.train()
    for offset in (3., 7., 12.):
        model(torch.arange(12, dtype=torch.float32).reshape(4, 3) + offset)
    ema.step(model)
    for name, value in model.named_buffers():
        torch.testing.assert_close(dict(ema.averaged_model.named_buffers())[name], value)
    model.eval()
    inputs = torch.randn(6, 3)
    torch.testing.assert_close(ema.averaged_model(inputs), model(inputs))


def test_buffer_copy_does_not_change_parameter_decay():
    model = torch.nn.Linear(2, 2)
    model.register_buffer("calibration", torch.tensor([2., 3.]))
    model.register_buffer("counter", torch.tensor(0, dtype=torch.int64))
    ema = EMAModel(copy.deepcopy(model), power=1., inv_gamma=1.)
    for step in range(5):
        previous = ema.averaged_model.weight.clone()
        with torch.no_grad():
            model.weight.fill_(step + 4.)
            model.calibration.fill_(step + 10.)
            model.counter.fill_(step + 1)
        expected_decay = ema.get_decay(ema.optimization_step)
        ema.step(model)
        torch.testing.assert_close(ema.averaged_model.weight,
                                  previous * expected_decay + model.weight * (1 - expected_decay))
        torch.testing.assert_close(ema.averaged_model.calibration, model.calibration)
        torch.testing.assert_close(ema.averaged_model.counter, model.counter)
