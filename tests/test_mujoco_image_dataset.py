import numpy as np
import pytest
import torch

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.mujoco_image_dataset import MujocoImageDataset


@pytest.fixture
def mujoco_data(tmp_path):
    n_steps = 14
    positions = np.arange(n_steps * 3, dtype=np.float32).reshape(n_steps, 3)
    quaternions = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n_steps, 1))
    poses = np.concatenate([positions, quaternions], axis=-1)
    widths = np.linspace(0.01, 0.08, n_steps, dtype=np.float32)[:, None]
    action_poses = poses.copy()
    action_poses[:, :3] += 0.5
    action_widths = widths + 0.01
    images = np.broadcast_to(
        np.arange(n_steps, dtype=np.uint8)[:, None, None, None, None] * 17,
        (n_steps, 1, 4, 4, 3),
    ).copy()
    data = {
        'robot_0_camera_images': images,
        'robot_0_tcp_xyz_wxyz': poses,
        'robot_0_gripper_width': widths,
        'action_0_tcp_xyz_wxyz': action_poses,
        'action_0_gripper_width': action_widths,
    }
    buffer = ReplayBuffer.create_empty_numpy()
    for start in (0, 7):
        buffer.add_episode({key: value[start:start + 7] for key, value in data.items()})
    path = str(tmp_path / 'mujoco.zarr')
    buffer.save_to_path(path)
    expected = {
        'agent_pos': torch.from_numpy(np.concatenate([poses, widths], axis=-1)),
        'action': torch.from_numpy(np.concatenate([action_poses, action_widths], axis=-1)),
    }
    return path, expected


@pytest.mark.parametrize('mode', ['limits', 'gaussian'])
@pytest.mark.parametrize('horizon', [1, 3, 7])
def test_normalizer_matches_sampled_observations(mujoco_data, mode, horizon):
    path, expected = mujoco_data
    dataset = MujocoImageDataset(
        path, horizon=horizon, pad_before=1, pad_after=1, val_ratio=0.5,
    )
    normalizer = dataset.get_normalizer(mode=mode)

    for key, values in expected.items():
        field = normalizer[key]
        assert field.params_dict['scale'].shape == (8,)
        stats = field.get_input_stats()
        for stat, want in {
            'min': values.amin(dim=0),
            'max': values.amax(dim=0),
            'mean': values.mean(dim=0),
            'std': values.std(dim=0),
        }.items():
            torch.testing.assert_close(stats[stat], want)
        if mode == 'limits':
            normalized = field.normalize(values)
            torch.testing.assert_close(normalized[:, -1].amin(), torch.tensor(-1.0))
            torch.testing.assert_close(normalized[:, -1].amax(), torch.tensor(1.0))

    for split in (dataset, dataset.get_validation_dataset()):
        assert len(split) > 0
        for index in range(len(split)):
            sample = split[index]
            observation = sample['obs']['agent_pos']
            assert observation.shape == (horizon, 8)
            # At horizon=7 the old 14-D normalizer can reshape these 56 values
            # without raising, so also verify values rather than only shape.
            normalized = normalizer.normalize({
                'agent_pos': observation,
                'action': sample['action'],
                'image': sample['obs']['image'],
            })
            for key, values in [('agent_pos', observation), ('action', sample['action'])]:
                assert normalized[key].shape == values.shape
                assert torch.isfinite(normalized[key]).all()
                recovered = normalizer[key].unnormalize(normalized[key])
                torch.testing.assert_close(recovered, values)
            torch.testing.assert_close(
                normalized['image'], 2 * sample['obs']['image'] - 1,
            )
