import pytest
import torch
from diffusion_policy.model.bet.libraries.loss_fn import FocalLoss

@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("spatial", [False, True])
def test_all_ignored_focal_loss_keeps_tensor_and_gradients(reduction, spatial):
    shape = (2, 3, 2, 2) if spatial else (4, 3)
    x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    yshape = (2, 2, 2) if spatial else (4,)
    y = torch.full(yshape, -100, dtype=torch.long)
    result = FocalLoss(gamma=2., reduction=reduction)(x, y)
    assert result.dtype == x.dtype
    assert result.device == x.device
    assert result.shape == ((0,) if reduction == "none" else ())
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x))


def test_ignored_nonfinite_logits_do_not_contaminate_zero_loss():
    x = torch.full((2, 3), float("nan"), requires_grad=True)
    result = FocalLoss(gamma=2.)(x, torch.full((2,), -100, dtype=torch.long))
    assert result.item() == 0.
    result.backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x))


def test_mixed_focal_loss_matches_independent_formula():
    x = torch.tensor([[.2, -.7, .9], [4., 2., 1.], [-.1, .3, .5]], requires_grad=True)
    y = torch.tensor([2, -100, 1])
    actual = FocalLoss(gamma=2.)(x, y)
    probability = torch.softmax(x[[0, 2]], dim=-1)[torch.arange(2), torch.tensor([2, 1])]
    expected = -((1 - probability)**2 * torch.log(probability)).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(x.grad[1], torch.zeros(3))
