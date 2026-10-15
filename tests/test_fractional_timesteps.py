import numpy as np
import pytest
import torch
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion

@pytest.fixture(params=["unet", "transformer"])
def model(request):
    torch.manual_seed(4)
    if request.param == "unet":
        value = ConditionalUnet1D(3, down_dims=[8, 16], diffusion_step_embed_dim=8)
    else:
        value = TransformerForDiffusion(3, 3, 4, n_layer=1, n_head=2, n_emb=8,
                                       p_drop_emb=0., p_drop_attn=0.)
    return value.eval()

@pytest.mark.parametrize("timestep", [.25, 1.5, -.75])
def test_fractional_python_timestep_matches_tensor_and_analytic_embedding(model, timestep):
    sample = torch.randn(2, 4, 3)
    embedding = model.diffusion_step_encoder[0] if isinstance(model, ConditionalUnet1D) else model.time_emb
    observed = []
    hook = embedding.register_forward_hook(lambda module, args, result: observed.append(result.detach().clone()))
    try:
        with torch.no_grad():
            scalar_result = model(sample, timestep)
            tensor_result = model(sample, torch.tensor(timestep))
    finally:
        hook.remove()
    frequency = np.exp(-np.arange(4) * np.log(10000.) / 3)
    phase = timestep * frequency
    expected = torch.from_numpy(np.concatenate([np.sin(phase), np.cos(phase)])).to(observed[0])
    torch.testing.assert_close(observed[0], expected[None].expand(2, -1), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(scalar_result, tensor_result, rtol=0., atol=0.)

@pytest.mark.parametrize("timestep", [0, 2, 7])
def test_integer_python_timestep_is_unchanged(model, timestep):
    sample = torch.randn(2, 4, 3)
    with torch.no_grad():
        torch.testing.assert_close(model(sample, timestep), model(sample, torch.tensor(timestep)),
                                   rtol=0., atol=0.)
