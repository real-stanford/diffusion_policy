import pytest
import torch
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator, KeypointMaskGenerator

@pytest.mark.parametrize("kind", ["lowdim", "keypoint_shared", "keypoint_independent"])
def test_unseeded_masks_follow_global_rng(kind):
    if kind == "lowdim":
        generator = LowdimMaskGenerator(2, 8, max_n_obs_steps=5, fix_obs_steps=False)
    else:
        generator = KeypointMaskGenerator(2, 2, max_n_obs_steps=5, fix_obs_steps=False,
                                         time_independent=kind == "keypoint_independent")
    shape = (64, 6, 10)
    torch.manual_seed(17)
    first = generator(shape)
    second = generator(shape)
    assert not torch.equal(first, second)
    torch.manual_seed(17)
    assert torch.equal(generator(shape), first)
    assert torch.equal(generator(shape), second)

@pytest.mark.parametrize("keypoints", [False, True])
def test_explicit_mask_seed_is_reproducible_and_isolated(keypoints):
    generator = (KeypointMaskGenerator(2, 2, fix_obs_steps=False) if keypoints else
                 LowdimMaskGenerator(2, 8, fix_obs_steps=False))
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    first = generator((32, 6, 10), seed=7)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.rand(100)
    assert torch.equal(first, generator((32, 6, 10), seed=7))
