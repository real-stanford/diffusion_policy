"""Non-finite validation metrics must not poison top-k checkpoint selection."""
from diffusion_policy.common import checkpoint_util as target
import math
from pathlib import Path
import pytest

@pytest.mark.parametrize('mode', ['min', 'max'])
@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf])
def test_nonfinite_metric_does_not_reserve_a_slot(tmp_path, mode, bad):
    cls = target.TopKCheckpointManager
    manager = cls(tmp_path, 'score', mode=mode, k=2, format_str='{epoch}.ckpt')
    assert manager.get_ckpt_path({'score': bad, 'epoch': 0}) is None
    assert manager.path_value_map == {}
    assert list(tmp_path.iterdir()) == []

@pytest.mark.parametrize('mode,best,worse', [('min', 1.0, 2.0), ('max', 2.0, 1.0)])
def test_bad_metric_cannot_displace_a_finite_checkpoint(tmp_path, mode, best, worse):
    cls = target.TopKCheckpointManager
    manager = cls(tmp_path, 'score', mode=mode, k=1, format_str='{epoch}.ckpt')
    checkpoint = Path(manager.get_ckpt_path({'score': best, 'epoch': 1}))
    checkpoint.write_bytes(b'preserved checkpoint')
    for bad in (math.nan, math.inf, -math.inf):
        assert manager.get_ckpt_path({'score': bad, 'epoch': 2}) is None
        assert checkpoint.read_bytes() == b'preserved checkpoint'
    assert manager.get_ckpt_path({'score': worse, 'epoch': 3}) is None

@pytest.mark.parametrize('mode,start,better', [('min', 2.0, 1.0), ('max', 1.0, 2.0)])
def test_finite_replacement_is_unchanged(tmp_path, mode, start, better):
    cls = target.TopKCheckpointManager
    manager = cls(tmp_path, 'score', mode=mode, k=1, format_str='{epoch}.ckpt')
    old = Path(manager.get_ckpt_path({'score': start, 'epoch': 1}))
    old.write_bytes(b'old')
    new = manager.get_ckpt_path({'score': better, 'epoch': 2})
    assert new == str(tmp_path / '2.ckpt')
    assert not old.exists()
    assert manager.path_value_map == {new: better}

def test_disabled_manager_needs_no_metrics(tmp_path):
    cls = target.TopKCheckpointManager
    assert cls(tmp_path, 'missing', k=0).get_ckpt_path({}) is None
