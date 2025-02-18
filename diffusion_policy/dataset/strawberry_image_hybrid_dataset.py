import h5py
import numpy as np
import torch
import copy
import cv2
from pathlib import Path
from typing import Dict
from torch.utils.data import Dataset

class StrawberryImageHybridDataset(Dataset):
    """Optimized Dataset for PA Arm Simulation stored in HDF5."""

    def __init__(
        self,
        hdf5_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        seed: int = 42,
        val_ratio: float = 0.0,
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed
        self.val_ratio = val_ratio

        # Open HDF5 file in read mode
        self.file = h5py.File(hdf5_path, "r")
        self.demo_keys = sorted(
            [key for key in self.file["data"].keys() if key.startswith("demo_")]
        )

        self.episode_dict = {}
        self.all_joint_positions = []

        # Define indices to extract (0, 1, 2, 3, 10 for a final size of 5)
        selected_indices = np.array([0, 1, 2, 3, 10])

        for demo in self.demo_keys:
            replicator_rgb = self.file[f"data/{demo}/obs/cam_end_effector"][:]  # Load all at once (T, H, W, 3)
            joint_positions = self.file[f"data/{demo}/obs/low_dim_states"][:, selected_indices]  # Extract elements (T, 5)

            # Store episode steps efficiently as structured NumPy arrays
            self.episode_dict[demo] = {
                "replicator_rgb": replicator_rgb,
                "joint_positions": joint_positions
            }

            # Store first joint position for normalization
            self.all_joint_positions.append(joint_positions[0])

        # Vectorized sequence extraction
        self.step_list = []
        for demo in self.episode_dict.values():
            T = len(demo["replicator_rgb"])
            self.step_list.extend([
                {"replicator_rgb": demo["replicator_rgb"][i:i + horizon], "joint_positions": demo["joint_positions"][i:i + horizon]}
                for i in range(T - horizon)
            ])
        self.step_list = np.array(self.step_list, dtype=object)  # Convert to NumPy array for faster indexing

        # Efficient train-validation split
        np.random.seed(seed)
        random_id_list = np.random.permutation(len(self.step_list))
        split_idx = int(len(self.step_list) * (1 - self.val_ratio))
        self.train_id_list, self.val_id_list = (
            random_id_list[:split_idx],
            random_id_list[split_idx:],
        )
        self.step_id_list = self.train_id_list  # Default to training set
        print("loading dataset")

    def get_validation_dataset(self):
        """Get a validation dataset without reloading everything."""
        val_set = copy.copy(self)
        val_set.step_id_list = self.val_id_list
        return val_set

    def __len__(self) -> int:
        return len(self.step_id_list)

    def _preprocess_images(self, images):
        """Vectorized image preprocessing (resizing, color conversion, normalization)."""
        images = np.array(
            [cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA) for img in images]
        )
        return np.moveaxis(images, -1, 1).astype(np.float32) / 255.0  # Normalize

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        step = self.step_list[self.step_id_list[idx]]

        # Batch process data
        joint_positions = step["joint_positions"].astype(np.float32)  # (T, 5)
        replicator_rgb = self._preprocess_images(step["replicator_rgb"])  # (T, 3, 96, 96)

        return {
            "obs": {
                "joint_positions": torch.from_numpy(joint_positions),  # (T, 5)
                "replicator_rgb": torch.from_numpy(replicator_rgb),  # (T, 3, 96, 96)
            },
            "action": torch.from_numpy(joint_positions),  # (T, 5)
        }


if __name__ == "__main__":
    dataset = StrawberryImageHybridDataset("/path/to/hdf5_file.hdf5", horizon=16)
    print(f"Dataset size: {len(dataset)}")
