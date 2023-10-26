import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import diffusion_policy.model.bet.latent_generators.latent_generator as latent_generator

from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.bet.libraries.loss_fn import FocalLoss, soft_cross_entropy

from typing import Optional, Tuple

class Transformer(latent_generator.AbstractLatentGenerator):
    def __init__(
        self,
        input_dim: int,
        num_bins: int,
        action_dim: int,
        horizon: int,
        focal_loss_gamma: float,
        offset_loss_scale: float,
        **kwargs
    ):
        super().__init__()
        self.model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=num_bins * (1 + action_dim),
            horizon=horizon,
            **kwargs
        )
        self.vocab_size = num_bins
        self.focal_loss_gamma = focal_loss_gamma
        self.offset_loss_scale = offset_loss_scale
        self.action_dim = action_dim
    
    def get_optimizer(self, **kwargs) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(**kwargs)
    
    def get_latent_and_loss(self, 
            obs_rep: torch.Tensor, 
            target_latents: torch.Tensor, 
            return_loss_components=True,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_latents, target_offsets = target_latents
        target_latents = target_latents.view(-1)
        criterion = FocalLoss(gamma=self.focal_loss_gamma)

        t = torch.tensor(0, device=self.model.device)
        output = self.model(obs_rep, t)
        logits = output[:, :, : self.vocab_size]
        offsets = output[:, :, self.vocab_size :]
        batch = logits.shape[0]
        seq = logits.shape[1]
        offsets = einops.rearrange(
            offsets,
            "N T (V A) -> (N T) V A",  # N = batch, T = seq
            V=self.vocab_size,
            A=self.action_dim,
        )
        # calculate (optionally soft) cross entropy and offset losses
        class_loss = criterion(logits.view(-1, logits.size(-1)), target_latents)
        # offset loss is only calculated on the target class
        # if soft targets, argmax is considered the target class
        selected_offsets = offsets[
            torch.arange(offsets.size(0)),
            target_latents.view(-1),
        ]
        offset_loss = self.offset_loss_scale * F.mse_loss(
            selected_offsets, target_offsets.view(-1, self.action_dim)
        )
        loss = offset_loss + class_loss
        logits = einops.rearrange(logits, "batch seq classes -> seq batch classes")
        offsets = einops.rearrange(
            offsets,
            "(N T) V A -> T N V A",  # ? N, T order? Anyway does not affect loss and training (might affect visualization)
            N=batch,
            T=seq,
        )
        return (
            (logits, offsets),
            loss,
            {"offset": offset_loss, "class": class_loss, "total": loss},
        )

    def generate_latents(
        self, obs_rep: torch.Tensor
    ) -> torch.Tensor:
        t = torch.tensor(0, device=self.model.device)
        output = self.model(obs_rep, t)
        logits = output[:, :, : self.vocab_size]
        offsets = output[:, :, self.vocab_size :]
        offsets = einops.rearrange(
            offsets,
            "N T (V A) -> (N T) V A",  # N = batch, T = seq
            V=self.vocab_size,
            A=self.action_dim,
        )

        probs = F.softmax(logits, dim=-1)
        batch, seq, choices = probs.shape
        # Sample from the multinomial distribution, one per row.
        sampled_data = torch.multinomial(probs.view(-1, choices), num_samples=1)
        sampled_data = einops.rearrange(
            sampled_data, "(batch seq) 1 -> batch seq 1", batch=batch, seq=seq
        )
        sampled_offsets = offsets[
            torch.arange(offsets.shape[0]), sampled_data.flatten()
        ].view(batch, seq, self.action_dim)
        return (sampled_data, sampled_offsets)
