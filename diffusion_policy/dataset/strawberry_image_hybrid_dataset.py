import copy
from typing import Tuple

import cv2
import h5py
import numpy as np
import torch
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

IMAGE_SIZE = (112, 112)
PAD_RATIO = 0.0


def process_image(
    image: np.ndarray,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    pad_ratio: float = PAD_RATIO,
) -> np.ndarray:
    """Perform center crop with padding based on the shorter dimension.

    Args:
        image: Input image as numpy array
        image_size: Tuple of (height, width)
        pad_ratio: Ratio to determine padding size (e.g., 0.2 for 20% padding)

    Returns:
        Cropped image as numpy array
    """
    height, width = image.shape[:2]
    shorter_dim = min(width, height)
    crop_size = int(shorter_dim * (1.0 - pad_ratio))

    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2

    cropped = image[start_y : start_y + crop_size, start_x : start_x + crop_size]
    resized = cv2.resize(cropped, image_size)
    return resized


class StrawberryImageHybridDataset(BaseImageDataset):
    """Dataset for Strawberry image hybrid data."""

    def __init__(
        self,
        hdf5_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes

        # Open HDF5 in read mode, keep reference
        self.file = h5py.File(hdf5_path, "r")

        # Collect all demo keys
        self.demo_keys = sorted([
            key for key in self.file["data"].keys() if key.startswith("demo_")
        ])

        # Store references instead of loading into memory
        self.episodes = []
        indices = [0, 1, 2, 3, 10]
        for demo in self.demo_keys:
            self.episodes.append({
                "cam_end_effector": self.file[f"data/{demo}/obs/cam_end_effector"],
                "low_dim_states": self.file[f"data/{demo}/obs/low_dim_states"][
                    :, indices
                ],
                "actions": self.file[f"data/{demo}/obs/low_dim_states"][:, indices],
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

    def get_normalizer(self, mode="limits", **kwargs):
        """
        Create a `LinearNormalizer` object with the required state_dict format.
        """
        normalizer = LinearNormalizer()

        # Collect data across all demos
        data = {
            "action": np.concatenate(
                [ep["actions"][:] for ep in self.episodes], axis=0
            ),
            "low_dim_states": np.concatenate(
                [ep["low_dim_states"][:] for ep in self.episodes], axis=0
            ),
        }

        # Fit normalizer to data
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        # Normalize images separately (0-255 -> 0-1)
        normalizer["cam_end_effector"] = get_image_range_normalizer()

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
            pad = np.repeat(x[-1:, ...], self.horizon - T, axis=0)
            x = np.concatenate([x, pad], axis=0)
        return x

    def _sample_to_data(self, idx):
        """
        Fetches a sample efficiently.
        """
        demo_id, start_idx = self.index_map[idx]
        episode = self.episodes[demo_id]

        end_idx = start_idx + self.horizon

        agent_pos = episode["low_dim_states"][start_idx:end_idx].astype(np.float32)
        image = episode["cam_end_effector"][start_idx:end_idx]
        action = episode["actions"][start_idx + 1 : end_idx + 1].astype(np.float32)

        agent_pos = self.pad_to_horizon(agent_pos)
        image = self.pad_to_horizon(image)
        action = self.pad_to_horizon(action)

        image = np.array([process_image(frame) for frame in image])
        image = np.moveaxis(image, -1, 1) / 255

        data = {
            "obs": {
                "cam_end_effector": image,
                "low_dim_states": agent_pos,
            },
            "action": action,
        }
        return data

    def __getitem__(self, idx):
        data = self._sample_to_data(idx)

        # Convert to Torch tensors
        torch_data = {
            "obs": {
                "cam_end_effector": torch.from_numpy(data["obs"]["cam_end_effector"]),
                "low_dim_states": torch.from_numpy(data["obs"]["low_dim_states"]),
            },
            "action": torch.from_numpy(data["action"]),
        }

        return torch_data


def try_dataset():
    dataset = StrawberryImageHybridDataset(
        horizon=16,
        max_train_episodes=90,
        pad_after=7,
        pad_before=1,
        seed=42,
        val_ratio=0.02,
        hdf5_path="../../data/output_50.hdf5",
    )

    import IPython

    IPython.embed(colors="neutral")
    exit(0)

    from matplotlib import pyplot as plt

    normalizer = dataset.get_normalizer()
    nactions = normalizer["action"].normalize(dataset.replay_buffer["action"])
    diff = np.diff(nactions, axis=0)
    dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)


if __name__ == "__main__":
    try_dataset()
