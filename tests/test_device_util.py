import os, sys, unittest
from unittest import mock
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import numpy as np
import torch
from diffusion_policy.common import device_util
from diffusion_policy.model.common.tensor_util import to_numpy, to_list


def fake_host(xpu=False, cuda=False):
    return mock.patch.object(
        device_util, 'is_available',
        side_effect=lambda t: {'xpu': xpu, 'cuda': cuda}.get(t, t == 'cpu'))


class TestDeviceDispatch(unittest.TestCase):
    def test_xpu_preferred_over_cuda(self):
        with fake_host(xpu=True, cuda=True):
            self.assertEqual(device_util.resolve_device(None), torch.device('xpu', 0))

    def test_cuda_when_no_xpu(self):
        with fake_host(cuda=True):
            self.assertEqual(device_util.resolve_device(None), torch.device('cuda', 0))

    def test_cpu_fallback(self):
        with fake_host():
            self.assertEqual(device_util.resolve_device(None), torch.device('cpu'))

    def test_explicit_device_wins(self):
        with fake_host(xpu=True):
            self.assertEqual(device_util.resolve_device('cuda:1'), torch.device('cuda', 1))
            self.assertEqual(device_util.resolve_device('cpu'), torch.device('cpu'))

    def test_eval_default_is_detection(self):
        src = open(os.path.join(ROOT_DIR, 'eval.py')).read()
        self.assertIn("'--device', default=None", src)


class TestToNumpyNonCpu(unittest.TestCase):
    def test_cpu_tensors(self):
        self.assertTrue(isinstance(to_numpy({'a': torch.ones(2)})['a'], np.ndarray))
        self.assertEqual(to_list({'a': torch.ones(2)}), {'a': [1.0, 1.0]})

    def test_non_cpu_tensor_is_copied_to_host(self):
        # a tensor reporting a non-cpu device (xpu or cuda) must go through .cpu()
        class FakeTensor(torch.Tensor):
            @property
            def device(self):
                return torch.device('meta')
            def cpu(self, *args, **kwargs):
                return torch.Tensor.cpu(self)
        t = torch.ones(2).as_subclass(FakeTensor)
        self.assertEqual(to_list(t), [1.0, 1.0])
        self.assertTrue(isinstance(to_numpy(t), np.ndarray))


@unittest.skipUnless(hasattr(torch, 'xpu') and torch.xpu.is_available(), 'needs an XPU')
class TestXpuParity(unittest.TestCase):
    def test_unet_lowdim_predict_action(self):
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy

        torch.manual_seed(0)
        model = ConditionalUnet1D(input_dim=2, global_cond_dim=8, down_dims=[32, 64],
            kernel_size=3, n_groups=4)
        for p in model.parameters():  # re-randomise: no zero-initialised layer may hide a mismatch
            torch.nn.init.normal_(p, std=0.1)
        scheduler = DDIMScheduler(num_train_timesteps=10, beta_schedule='squaredcos_cap_v2',
            clip_sample=False, prediction_type='epsilon')
        policy = DiffusionUnetLowdimPolicy(model, scheduler, horizon=8, obs_dim=4, action_dim=2,
            n_action_steps=4, n_obs_steps=2, obs_as_global_cond=True, num_inference_steps=5)
        normalizer = LinearNormalizer()
        normalizer.fit({'obs': torch.randn(16, 4), 'action': torch.randn(16, 2)})
        policy.set_normalizer(normalizer)
        policy.eval()
        obs = torch.randn(1, 2, 4)
        noise = torch.randn(1, 8, 2)  # drawn on CPU, shared by both devices

        outs = []
        for dev in ('cpu', 'xpu:0'):
            policy.to(dev)
            real_randn = torch.randn
            def randn(*args, **kwargs):
                kwargs.pop('generator', None)
                if kwargs.get('size') == tuple(noise.shape):
                    return noise.to(kwargs.get('device', 'cpu'), kwargs.get('dtype', torch.float32))
                return real_randn(*args, **kwargs)
            with mock.patch.object(torch, 'randn', randn), torch.no_grad():
                outs.append(policy.predict_action({'obs': obs.to(dev)})['action'].cpu())
        torch.testing.assert_close(outs[0], outs[1], atol=1e-3, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
