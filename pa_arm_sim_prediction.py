"""
Copyright 2025 Zordi, Inc. All rights reserved.

Usage:
python pa_arm_sim_prediction.py -c checkpoint_pa_arm_sim_2025_0217.ckpt
"""

import click
import dill
import hydra
import torch
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace


class ActionModel:
    def __init__(self, checkpoint_filepath: str):
        payload = torch.load(open(checkpoint_filepath, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        self.policy: BaseImagePolicy
        self.policy = workspace.model
        if cfg.training.use_ema:
            self.policy = workspace.ema_model

        self.device = torch.device("cuda")
        self.policy.eval().to(self.device)

    def predict(
        self, images: torch.Tensor, joint_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            images: torch.Tensor, shape: (2, 3, 96, 96)
            joint_positions: torch.Tensor, shape: (2, 4)

        Returns:
            action: torch.Tensor, shape: (8, 4)
        """
        obs_dict = {
            "replicator_rgb": images.unsqueeze(0).to(self.device),
            "joint_positions": joint_positions.unsqueeze(0).to(self.device),
        }
        action = self.policy.predict_action(obs_dict)
        return action["action"].squeeze().detach().cpu()


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output", required=False, default="./pa_arm_sim_output")
def main(checkpoint, output):
    # load checkpoint
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device("cuda")
    policy.eval().to(device)

    obs_dict = {
        "replicator_rgb": torch.randn(2, 3, 96, 96).unsqueeze(0).to(device),
        "joint_positions": torch.randn(2, 4).unsqueeze(0).to(device),
    }

    action = policy.predict_action(obs_dict)
    print(action)
    print(action["action"].squeeze().shape)

    print("Done.")


if __name__ == "__main__":
    main()
