import numpy as np
import pytest
import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer, get_optimal_chunks

@pytest.mark.parametrize("backend", ["numpy", "zarr"])
def test_reconstruct_empty_buffer_after_dropping_last_episode(backend):
    buffer = (ReplayBuffer.create_empty_numpy() if backend == "numpy" else
              ReplayBuffer.create_empty_zarr())
    buffer.add_episode({"obs": np.ones((3, 2), dtype=np.float32)})
    buffer.drop_episode()
    restored = ReplayBuffer(buffer.root)
    assert restored.n_steps == restored.n_episodes == 0
    assert restored["obs"].shape == (0, 2)
    restored.add_episode({"obs": np.full((2, 2), 7., dtype=np.float32)})
    np.testing.assert_array_equal(restored.get_episode(0)["obs"], np.full((2, 2), 7.))


def test_empty_buffer_store_round_trip_preserves_field_schema():
    buffer = ReplayBuffer.create_empty_numpy()
    buffer.add_episode({"obs": np.ones((3, 2), dtype=np.float32)})
    buffer.drop_episode()
    store = zarr.MemoryStore()
    buffer.save_to_store(store)
    restored = ReplayBuffer.copy_from_store(store)
    assert restored.n_steps == restored.n_episodes == 0
    assert restored["obs"].shape == (0, 2)


def test_nonempty_data_without_episode_boundaries_remains_invalid():
    root = {"data": {"obs": np.ones((1, 2))},
            "meta": {"episode_ends": np.empty(0, dtype=np.int64)}}
    with pytest.raises(AssertionError):
        ReplayBuffer(root)


@pytest.mark.parametrize("shape", [(0,), (0, 2), (0, 3, 2), (0, 0)])
def test_empty_array_chunks_remain_positive(shape):
    chunks = get_optimal_chunks(shape, np.float32)
    assert len(chunks) == len(shape)
    assert all(size > 0 for size in chunks)
