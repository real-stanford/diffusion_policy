import os
import h5py
import torch
import numpy as np
import copy
from typing import Dict
from diffusion_policy.common.pytorch_util import dict_apply

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer




import cv2

from torch.utils.data import Dataset
print(cv2.__version__)                                  




class StrawberryImageHybridDataset(BaseImageDataset):
    def __init__(self,
                 hdf5_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0):
        super().__init__()
        self.hdf5_path = hdf5_path

        self.horizon =horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        # Open HDF5 in read mode, keep reference
        self.file = h5py.File(hdf5_path, "r")

        # Collect all demo keys
        self.demo_keys = sorted([key for key in self.file["data"].keys() if key.startswith("demo_")])

        # Store references instead of loading into memory
        self.episodes = []
        for demo in self.demo_keys:
            self.episodes.append({
                "actions": self.file[f"data/{demo}/actions"],  # Shape (T, 8)
                "cam_end_effector": self.file[f"data/{demo}/obs/cam_end_effector"],  # Shape (T, 480, 640, 3)
                "low_dim_states": self.file[f"data/{demo}/obs/low_dim_states"]  # Shape (T, 12)
            })

        # Compute total number of samples efficiently
        self.index_map = []
        for demo_id, episode in enumerate(self.episodes):
            T = len(episode["actions"])
            for i in range(T - horizon):  # Only valid sequences
                self.index_map.append((demo_id, i))

        self.n_samples = len(self.index_map)

        # Create train-validation mask
        np.random.seed(seed)
        val_mask = np.random.rand(self.n_samples) < val_ratio
        self.train_mask = ~val_mask

        # Initialize normalizer
        self.normalizer = self.get_normalizer()

    def get_normalizer(self, mode="limits", **kwargs):
        """
        Create a `LinearNormalizer` object with the required state_dict format.
        """
        normalizer = LinearNormalizer()

        # Collect data across all demos
        data = {
            'action': np.concatenate([ep["actions"][:] for ep in self.episodes], axis=0),  # (N, 8)
            'low_dim_states': np.concatenate([ep["low_dim_states"][:] for ep in self.episodes], axis=0)  # (N, 12)
        }

        # Fit normalizer to data
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        # Normalize images separately (0-255 -> 0-1)
        normalizer['cam_end_effector'] = get_image_range_normalizer()

        return normalizer

    def get_validation_dataset(self):
        """Creates a validation dataset copy without reloading everything."""
        val_set = copy.copy(self)
        val_set.train_mask = ~self.train_mask
        return val_set

    def __len__(self):
        return self.n_samples

    def pad_to_horizon(self, x):
        """
        Pads the sequence along axis 0 (time) so that its length equals 'horizon'.
        If x has fewer than 'horizon' timesteps, the last timestep is repeated.
        """
        T = x.shape[0]
        if T < self.horizon:
            pad = np.repeat(x[-1:,...], self.horizon - T, axis=0)
            x = np.concatenate([x, pad], axis=0)
        return x


    def _sample_to_data(self, idx):
        """
        Fetches a sample efficiently.
        """
        demo_id, start_idx = self.index_map[idx]
        episode = self.episodes[demo_id]

        end_idx = start_idx + self.horizon

        agent_pos = episode["low_dim_states"][start_idx:end_idx].astype(np.float32)  # (T, 12)
        image = episode["cam_end_effector"][start_idx:end_idx]  # (T, 480, 640, 3)
        action = episode["actions"][start_idx:end_idx].astype(np.float32)  # (T, 8)

        agent_pos = self.pad_to_horizon(agent_pos)
        image = self.pad_to_horizon(image)
        action = self.pad_to_horizon(action)

        image = np.array([cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA) for frame in image])
        image = np.moveaxis(image, -1, 1) / 255


        data = {
            'obs': {
                'cam_end_effector': image, #Now T, 3, 96, 96
                'low_dim_states': agent_pos,  # Now (T, 12)
            },
            'action': action  # Now (T, 8)
        }
        return data

    def __getitem__(self, idx):
        data = self._sample_to_data(idx)

        # Convert to Torch tensors

        torch_data = {
            'obs': {
                'cam_end_effector': torch.from_numpy(data['obs']['cam_end_effector']),  # (T, 3, 96, 96)
                'low_dim_states': torch.from_numpy(data['obs']['low_dim_states']),  # (T, 12)
            },
            'action': torch.from_numpy(data['action']),  # (T, 8)
        }


        return torch_data


# ========================== TEST FUNCTION ==========================
def test():
    hdf5_path = "/home/zordi/robomimic/robomimic/scripts/output_50.hdf5"  # Change this path accordingly
    dataset = StrawberryImageHybridDataset(hdf5_path, horizon=16)
    print(f"Dataset size: {len(dataset)}")


if __name__ == "__main__":
    test()
