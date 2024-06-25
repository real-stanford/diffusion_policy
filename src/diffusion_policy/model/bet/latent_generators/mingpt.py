import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import diffusion_policy.model.bet.latent_generators.latent_generator as latent_generator

import diffusion_policy.model.bet.libraries.mingpt.model as mingpt_model
import diffusion_policy.model.bet.libraries.mingpt.trainer as mingpt_trainer
from diffusion_policy.model.bet.libraries.loss_fn import FocalLoss, soft_cross_entropy

from typing import Optional, Tuple


class MinGPT(latent_generator.AbstractLatentGenerator):
    def __init__(
        self,
        input_dim: int,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        block_size: int = 128,
        vocab_size: int = 50257,
        latent_dim: int = 768,  # Ignore, used for compatibility with other models.
        action_dim: int = 0,
        discrete_input: bool = False,
        predict_offsets: bool = False,
        offset_loss_scale: float = 1.0,
        focal_loss_gamma: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.action_dim = action_dim
        self.predict_offsets = predict_offsets
        self.offset_loss_scale = offset_loss_scale
        self.focal_loss_gamma = focal_loss_gamma
        for k, v in kwargs.items():
            setattr(self, k, v)

        gpt_config = mingpt_model.GPTConfig(
            input_size=self.input_size,
            vocab_size=self.vocab_size * (1 + self.action_dim)
            if self.predict_offsets
            else self.vocab_size,
            block_size=self.block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            discrete_input=discrete_input,
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
        )

        self.model = mingpt_model.GPT(gpt_config)

    def get_latent_and_loss(
        self,
        obs_rep: torch.Tensor,
        target_latents: torch.Tensor,
        seq_masks: Optional[torch.Tensor] = None,
        return_loss_components: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Unlike torch.transformers, GPT takes in batch x seq_len x embd_dim
        # obs_rep = einops.rearrange(obs_rep, "seq batch embed -> batch seq embed")
        # target_latents = einops.rearrange(
        #     target_latents, "seq batch embed -> batch seq embed"
        # )
        # While this has been trained autoregressively,
        # there is no reason why it needs to be so.
        # We can just use the observation as the input and the next latent as the target.
        if self.predict_offsets:
            target_latents, target_offsets = target_latents
        is_soft_target = (target_latents.shape[-1] == self.vocab_size) and (
            self.vocab_size != 1
        )
        if is_soft_target:
            target_latents = target_latents.view(-1, target_latents.size(-1))
            criterion = soft_cross_entropy
        else:
            target_latents = target_latents.view(-1)
            if self.vocab_size == 1:
                # unify k-means (target_class == 0) and GMM (target_prob == 1)
                target_latents = torch.zeros_like(target_latents)
            criterion = FocalLoss(gamma=self.focal_loss_gamma)
        if self.predict_offsets:
            output, _ = self.model(obs_rep)
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
                target_latents.argmax(dim=-1).view(-1)
                if is_soft_target
                else target_latents.view(-1),
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
            if return_loss_components:
                return (
                    (logits, offsets),
                    loss,
                    {"offset": offset_loss, "class": class_loss, "total": loss},
                )
            else:
                return (logits, offsets), loss
        else:
            logits, _ = self.model(obs_rep)
            loss = criterion(logits.view(-1, logits.size(-1)), target_latents)
            logits = einops.rearrange(
                logits, "batch seq classes -> seq batch classes"
            )  # ? N, T order? Anyway does not affect loss and training (might affect visualization)
            if return_loss_components:
                return logits, loss, {"class": loss, "total": loss}
            else:
                return logits, loss

    def generate_latents(
        self, obs_rep: torch.Tensor
    ) -> torch.Tensor:
        batch, seq, embed = obs_rep.shape

        output, _ = self.model(obs_rep, None)
        if self.predict_offsets:
            logits = output[:, :, : self.vocab_size]
            offsets = output[:, :, self.vocab_size :]
            offsets = einops.rearrange(
                offsets,
                "N T (V A) -> (N T) V A",  # N = batch, T = seq
                V=self.vocab_size,
                A=self.action_dim,
            )
        else:
            logits = output
        probs = F.softmax(logits, dim=-1)
        batch, seq, choices = probs.shape
        # Sample from the multinomial distribution, one per row.
        sampled_data = torch.multinomial(probs.view(-1, choices), num_samples=1)
        sampled_data = einops.rearrange(
            sampled_data, "(batch seq) 1 -> batch seq 1", batch=batch, seq=seq
        )
        if self.predict_offsets:
            sampled_offsets = offsets[
                torch.arange(offsets.shape[0]), sampled_data.flatten()
            ].view(batch, seq, self.action_dim)

            return (sampled_data, sampled_offsets)
        else:
            return sampled_data

    def get_optimizer(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        trainer_cfg = mingpt_trainer.TrainerConfig(
            weight_decay=weight_decay, learning_rate=learning_rate, betas=betas
        )
        return self.model.configure_optimizers(trainer_cfg)
