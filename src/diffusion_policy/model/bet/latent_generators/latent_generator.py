import abc
import torch
from typing import Tuple, Optional

import diffusion_policy.model.bet.utils as utils


class AbstractLatentGenerator(abc.ABC, utils.SaveModule):
    """
    Abstract class for a generative model that can generate latents given observation representations.

    In the probabilisitc sense, this model fits and samples from P(latent|observation) given some observation.
    """

    @abc.abstractmethod
    def get_latent_and_loss(
        self,
        obs_rep: torch.Tensor,
        target_latents: torch.Tensor,
        seq_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a set of observation representation and generated latents, get the encoded latent and the loss.

        Inputs:
        input_action: Batch of the actions taken in the multimodal demonstrations.
        target_latents: Batch of the latents that the generator should learn to generate the actions from.
        seq_masks: Batch of masks that indicate which timesteps are valid.

        Outputs:
        latent: The sampled latent from the observation.
        loss: The loss of the latent generator.
        """
        pass

    @abc.abstractmethod
    def generate_latents(
        self, seq_obses: torch.Tensor, seq_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Given a batch of sequences of observations, generate a batch of sequences of latents.

        Inputs:
        seq_obses: Batch of sequences of observations, of shape seq x batch x dim, following the transformer convention.
        seq_masks: Batch of sequences of masks, of shape seq x batch, following the transformer convention.

        Outputs:
        seq_latents: Batch of sequences of latents of shape seq x batch x latent_dim.
        """
        pass

    def get_optimizer(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        """
        Default optimizer class. Override this if you want to use a different optimizer.
        """
        return torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas
        )


class LatentGeneratorDataParallel(torch.nn.DataParallel):
    def get_latent_and_loss(self, *args, **kwargs):
        return self.module.get_latent_and_loss(*args, **kwargs)  # type: ignore

    def generate_latents(self, *args, **kwargs):
        return self.module.generate_latents(*args, **kwargs)  # type: ignore

    def get_optimizer(self, *args, **kwargs):
        return self.module.get_optimizer(*args, **kwargs)  # type: ignore
