"""
Copyright 2025 Zordi, Inc. All rights reserved.

Usage:
python pa_arm_sim_prediction.py -c checkpoint_pa_arm_robot_2025_0218.ckpt
"""

import click
import dill
import hydra
import torch
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace


class ActionModel:
    """Predict action from images and joint positions.

    Args:
        checkpoint_file_path: path to the checkpoint file
    """

    def __init__(self, checkpoint_file_path: str):
        payload = torch.load(open(checkpoint_file_path, "rb"), pickle_module=dill)
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
            images: torch.Tensor, shape: e.g. (obs_len, C, H, W)
            joint_positions: torch.Tensor, shape: e.g. (obs_len, jnt_gim)

        Returns:
            action: torch.Tensor, shape: e.g. (act_len, jnt_gim)
        """
        obs_dict = {
            "images": images.unsqueeze(0).to(self.device),
            "joint_positions": joint_positions.unsqueeze(0).to(self.device),
        }
        action = self.policy.predict_action(obs_dict)
        return action["action"].squeeze().detach().cpu()


@click.command()
@click.option("-c", "--checkpoint", required=True)
def main(checkpoint: str) -> None:  # noqa: D103
    # load checkpoint
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # configure policy
    policy: BaseImagePolicy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device("cuda")
    policy.eval().to(device)

    # try the policy with random input
    obs_dict = {
        "images": torch.randn(2, 3, 96, 96).unsqueeze(0).to(device),
        "joint_positions": torch.randn(2, 4).unsqueeze(0).to(device),
    }

    action = policy.predict_action(obs_dict)
    print(action)
    print(action["action"].squeeze().shape)

    print("Done.")


if __name__ == "__main__":
    main()
