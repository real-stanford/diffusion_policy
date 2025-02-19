import copy
import os
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import orjson
import torch
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class PAArmSimDataset(BaseImageDataset):
    """Dataset for PA arm simulation data."""

    def __init__(
        self,
        dataset_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
    ):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes

        self.all_joint_positions = list()

        self.episode_dict = dict()
        for episode_path in self.dataset_path.glob("*"):
            ep_name = episode_path.name
            ep_logs = orjson.loads((episode_path / "joint_state.json").read_bytes())
            ep_list = [
                {
                    "episode_name": ep_name,
                    "joint_positions": action_log["data"]["joint_positions"],
                    "images": action_log["replicator_rgb"],
                }
                for action_log in ep_logs["Isaac Sim Data"]
            ]
            self.episode_dict[episode_path.name] = ep_list
            self.all_joint_positions.append(np.array(ep_list[0]["joint_positions"]))

        self.step_list = list()
        for ep_list in self.episode_dict.values():
            for idx in range(len(ep_list) - horizon):
                self.step_list.append(ep_list[idx : idx + horizon])

        split_idx = int(len(self.step_list) * (1 - self.val_ratio))
        random_id_list = np.random.RandomState(self.seed).permutation(
            np.arange(len(self.step_list))
        )
        self.train_id_list = random_id_list[:split_idx]
        self.val_id_list = random_id_list[split_idx:]

        self.step_id_list = self.train_id_list

    def get_validation_dataset(self):
        """Get a validation dataset."""
        val_set = copy.copy(self)
        val_set.step_id_list = self.val_id_list
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        """Get a normalizer for the dataset."""
        normalizer = LinearNormalizer()
        normalizer["joint_positions"] = get_range_normalizer_from_stat(
            array_to_stats(np.array(self.all_joint_positions))
        )
        normalizer["action"] = normalizer["joint_positions"]
        normalizer["images"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.step_id_list)

    def _load_image(self, episode_name, image_path):
        img = cv2.imread(os.path.join(self.dataset_path, episode_name, image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.moveaxis(img, -1, 0)
        img = img.astype(np.float32)
        img /= 255
        return img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        step = self.step_list[self.step_id_list[idx]]

        joint_positions = list()
        rgb_images = list()
        for i in range(len(step)):
            joint_positions.append(
                np.array(step[i]["joint_positions"], dtype=np.float32)
            )
            rgb_images.append(
                self._load_image(step[i]["episode_name"], step[i]["images"])
            )

        torch_data = {
            "obs": {
                "joint_positions": torch.from_numpy(np.array(joint_positions)).float(),
                "images": torch.from_numpy(np.array(rgb_images)).float(),
            },
            "action": torch.from_numpy(np.array(joint_positions)).float(),
        }
        return torch_data


def try_dataset():
    dataset = PAArmSimDataset(
        horizon=16,
        max_train_episodes=90,
        pad_after=7,
        pad_before=1,
        seed=42,
        val_ratio=0.02,
        dataset_path="../../data/pa_arm_sim",
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
