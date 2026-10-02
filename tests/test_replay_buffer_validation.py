"""Episode-validation regressions on both in-memory storage backends."""

import numpy as np
import pytest
from diffusion_policy.common.replay_buffer import ReplayBuffer


@pytest.fixture(params=["numpy", "zarr"])
def buffer(request):
    if request.param == "numpy":
        return ReplayBuffer.create_empty_numpy()
    return ReplayBuffer.create_empty_zarr()


def first_episode():
    return {
        "action": np.array([[0.0], [1.0]], dtype=np.float32),
        "obs": np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
    }


def snapshot(buffer):
    return {key: value[:].copy() for key, value in buffer.items()}, buffer.episode_ends[
        :
    ].copy()


def assert_unchanged(buffer, state):
    data, ends = state
    assert set(buffer.keys()) == set(data)
    np.testing.assert_array_equal(buffer.episode_ends[:], ends)
    for key, value in data.items():
        np.testing.assert_array_equal(buffer[key][:], value)
        assert buffer[key].shape[0] == buffer.n_steps
    # The constructor independently checks the temporal-length invariant.
    ReplayBuffer(buffer.root)


def test_missing_existing_field_does_not_corrupt_buffer(buffer):
    buffer.add_episode(first_episode())
    state = snapshot(buffer)
    with pytest.raises(ValueError, match="missing existing data keys"):
        buffer.add_episode({"action": np.array([[10.0]], dtype=np.float32)})
    assert_unchanged(buffer, state)
    buffer.add_episode(
        {
            "action": np.array([[10.0]], dtype=np.float32),
            "obs": np.array([[11.0, 12.0]], dtype=np.float32),
        }
    )
    np.testing.assert_array_equal(buffer.episode_ends[:], [2, 3])


def test_late_shape_error_does_not_resize_earlier_fields(buffer):
    buffer.add_episode(first_episode())
    state = snapshot(buffer)
    # action is valid and is visited before the incompatible obs field.
    with pytest.raises(ValueError, match="incompatible sample shape"):
        buffer.add_episode(
            {
                "action": np.array([[10.0]], dtype=np.float32),
                "obs": np.array([[11.0]], dtype=np.float32),
            }
        )
    assert_unchanged(buffer, state)


def test_new_field_is_not_created_before_late_shape_error(buffer):
    buffer.add_episode(first_episode())
    state = snapshot(buffer)
    with pytest.raises(ValueError, match="incompatible sample shape"):
        buffer.add_episode(
            {
                "camera": np.ones((1, 2), dtype=np.uint8),
                "action": np.array([[10.0]], dtype=np.float32),
                "obs": np.array([[11.0]], dtype=np.float32),
            }
        )
    assert_unchanged(buffer, state)


def test_empty_episode_rejected_before_creating_fields(buffer):
    state = snapshot(buffer)
    with pytest.raises(ValueError, match="at least one step"):
        buffer.add_episode({"action": np.empty((0, 1), dtype=np.float32)})
    assert_unchanged(buffer, state)


def test_empty_episode_rejected_in_populated_buffer(buffer):
    buffer.add_episode(first_episode())
    state = snapshot(buffer)
    with pytest.raises(ValueError, match="at least one step"):
        buffer.add_episode(
            {
                "action": np.empty((0, 1), dtype=np.float32),
                "obs": np.empty((0, 2), dtype=np.float32),
            }
        )
    assert_unchanged(buffer, state)


def test_inconsistent_lengths_do_not_mutate_buffer(buffer):
    buffer.add_episode(first_episode())
    state = snapshot(buffer)
    with pytest.raises(ValueError, match="same number of steps"):
        buffer.add_episode(
            {
                "action": np.ones((2, 1), dtype=np.float32),
                "obs": np.ones((1, 2), dtype=np.float32),
            }
        )
    assert_unchanged(buffer, state)


def test_scalar_field_rejected_before_mutation(buffer):
    state = snapshot(buffer)
    with pytest.raises(ValueError, match="time dimension"):
        buffer.add_episode({"action": np.array(1.0)})
    assert_unchanged(buffer, state)


def test_no_fields_rejected(buffer):
    state = snapshot(buffer)
    with pytest.raises(ValueError, match="at least one data array"):
        buffer.add_episode({})
    assert_unchanged(buffer, state)


def test_new_fields_still_receive_historical_zero_padding(buffer):
    buffer.add_episode(first_episode())
    buffer.add_episode(
        {
            "action": np.array([[10.0]], dtype=np.float32),
            "obs": np.array([[11.0, 12.0]], dtype=np.float32),
            "camera": np.array([[13, 14]], dtype=np.uint8),
        }
    )
    np.testing.assert_array_equal(buffer["camera"][:], [[0, 0], [0, 0], [13, 14]])
    np.testing.assert_array_equal(buffer.get_episode(1)["action"], [[10.0]])
    np.testing.assert_array_equal(buffer.episode_ends[:], [2, 3])
    ReplayBuffer(buffer.root)


def test_valid_append_pop_and_reappend_preserve_alignment(buffer):
    buffer.add_episode(first_episode())
    next_episode = {
        "action": np.array([[10.0]], dtype=np.float32),
        "obs": np.array([[11.0, 12.0]], dtype=np.float32),
    }
    buffer.add_episode(next_episode)
    popped = buffer.pop_episode()
    for key, values in next_episode.items():
        np.testing.assert_array_equal(popped[key], values)
        np.testing.assert_array_equal(buffer[key][:], first_episode()[key])
    buffer.add_episode(next_episode)
    for values in buffer.values():
        assert values.shape[0] == buffer.n_steps
    np.testing.assert_array_equal(buffer.episode_lengths, [2, 1])
