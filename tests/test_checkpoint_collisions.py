"""Preserve checkpoint slots and files when formatted paths collide."""

from pathlib import Path
import pytest
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager


@pytest.mark.parametrize(
    "mode, first, better, worse, other",
    [
        ("min", 1.04, 1.02, 1.049, 2.0),
        ("max", 2.01, 2.04, 2.001, 1.0),
    ],
)
@pytest.mark.parametrize("capacity", [1, 2, 3])
def test_collision_improves_only_its_existing_slot(
    tmp_path, mode, first, better, worse, other, capacity
):
    manager = TopKCheckpointManager(
        tmp_path, "score", mode=mode, k=capacity, format_str="score={score:.1f}.ckpt"
    )
    first_path = manager.get_ckpt_path({"score": first})
    Path(first_path).write_bytes(b"existing best model")
    if capacity == 2:
        other_path = manager.get_ckpt_path({"score": other})
        Path(other_path).write_bytes(b"other selected model")
    before = manager.path_value_map.copy()
    candidate = manager.get_ckpt_path({"score": better})
    assert candidate == first_path
    expected = before | {first_path: better}
    assert manager.path_value_map == expected
    # Selection must not remove the old file before the caller saves its replacement.
    assert Path(first_path).read_bytes() == b"existing best model"
    for path in before:
        assert Path(path).exists()
    Path(candidate).write_bytes(b"replacement best model")
    updated = manager.path_value_map.copy()
    assert manager.get_ckpt_path({"score": worse}) is None
    assert manager.get_ckpt_path({"score": better}) is None
    assert manager.path_value_map == updated
    assert Path(first_path).read_bytes() == b"replacement best model"


@pytest.mark.parametrize(
    "mode, scores, improved",
    [
        ("min", [1.0, 2.0], 0.5),
        ("max", [2.0, 1.0], 3.0),
    ],
)
def test_distinct_paths_still_evict_the_worst_checkpoint(
    tmp_path, mode, scores, improved
):
    manager = TopKCheckpointManager(
        tmp_path, "score", mode=mode, k=2, format_str="score={score:.1f}.ckpt"
    )
    paths = []
    for score in scores:
        path = manager.get_ckpt_path({"score": score})
        Path(path).write_bytes(str(score).encode())
        paths.append(path)
    candidate = manager.get_ckpt_path({"score": improved})
    assert candidate is not None
    assert manager.path_value_map == {paths[0]: scores[0], candidate: improved}
    assert Path(paths[0]).exists()
    assert not Path(paths[1]).exists()


def test_zero_capacity_skips_candidates(tmp_path):
    manager = TopKCheckpointManager(tmp_path, "score", k=0)
    assert manager.get_ckpt_path({"score": 1.0}) is None
    assert manager.path_value_map == {}
