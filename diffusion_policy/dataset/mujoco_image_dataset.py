from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class MujocoImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            # zarr_path, keys=['img', 'state', 'action'])
            zarr_path, keys=['robot_0_camera_images', 'robot_0_tcp_xyz_wxyz', 'robot_0_gripper_width', 'action_0_tcp_xyz_wxyz', 'action_0_gripper_width'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': np.concatenate([self.replay_buffer['action_0_tcp_xyz_wxyz'], self.replay_buffer['action_0_gripper_width']], axis=-1),
            'agent_pos': np.concatenate([self.replay_buffer['robot_0_tcp_xyz_wxyz'], self.replay_buffer['robot_0_tcp_xyz_wxyz']], axis=-1)
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        agent_pos = np.concatenate([sample['robot_0_tcp_xyz_wxyz'], sample['robot_0_gripper_width']], axis=-1).astype(np.float32)
        agent_action = np.concatenate([sample['action_0_tcp_xyz_wxyz'], sample['action_0_gripper_width']], axis=-1).astype(np.float32)
        # image = np.moveaxis(sample['img'],-1,1)/255
        image = np.moveaxis(sample['robot_0_camera_images'].astype(np.float32).squeeze(1),-1,1)/255

        data = {
            'obs': {
                'image': image, # T, 3, 224, 224
                'agent_pos': agent_pos, # T, 8 (x,y,z,qx,qy,qz,qw,gripper_width)
            },
            'action': agent_action # T, 8 (x,y,z,qx,qy,qz,qw,gripper_width)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('/home/yihuai/robotics/repositories/mujoco/mujoco-env/data/collect_heuristic_data/2024-12-24_11-36-15_100episodes/merged_data.zarr')
    dataset = MujocoImageDataset(zarr_path, horizon=16)
    print(dataset[0])
    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)

if __name__ == '__main__':
    test()