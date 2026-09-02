import torch
from torch import nn

from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder


def _shape_meta():
    return {
        "obs": {
            "front": {"shape": (3, 8, 8), "type": "rgb"},
            "wrist": {"shape": (3, 8, 8), "type": "rgb"},
        }
    }


def _backbone():
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=1, bias=False),
        nn.BatchNorm2d(32),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )


def test_shared_backbone_replaces_batchnorm_with_groupnorm():
    """Shared RGB models must use per-sample normalization when requested."""
    torch.manual_seed(0)
    encoder = MultiImageObsEncoder(
        shape_meta=_shape_meta(),
        rgb_model=_backbone(),
        use_group_norm=True,
        share_rgb_model=True,
    ).train()

    shared_model = encoder.key_model_map["rgb"]
    assert not any(isinstance(module, nn.BatchNorm2d) for module in shared_model.modules())
    group_norms = [module for module in shared_model.modules() if isinstance(module, nn.GroupNorm)]
    assert len(group_norms) == 1
    assert group_norms[0].num_groups == 2
    assert group_norms[0].num_channels == 32

    front = torch.rand(1, 3, 8, 8)
    wrist = torch.rand(1, 3, 8, 8)
    single = encoder({"front": front, "wrist": wrist})
    batched = encoder(
        {
            "front": torch.cat((front, torch.rand(1, 3, 8, 8)), dim=0),
            "wrist": torch.cat((wrist, torch.rand(1, 3, 8, 8)), dim=0),
        }
    )

    assert single.shape == (1, 64)
    assert batched.shape == (2, 64)
    torch.testing.assert_close(single[0], batched[0], rtol=1e-5, atol=1e-6)
