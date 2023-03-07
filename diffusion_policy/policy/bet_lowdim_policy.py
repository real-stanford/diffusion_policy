from typing import Dict, Tuple
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.bet.action_ae.discretizers.k_means import KMeansDiscretizer
from diffusion_policy.model.bet.latent_generators.mingpt import MinGPT
from diffusion_policy.model.bet.utils import eval_mode

class BETLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            action_ae: KMeansDiscretizer, 
            obs_encoding_net: nn.Module, 
            state_prior: MinGPT,
            horizon,
            n_action_steps,
            n_obs_steps):
        super().__init__()
    
        self.normalizer = LinearNormalizer()
        self.action_ae = action_ae
        self.obs_encoding_net = obs_encoding_net
        self.state_prior = state_prior
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        T = self.horizon

        # pad To to T
        obs = torch.full((B,T,Do), -2, dtype=nobs.dtype, device=nobs.device)
        obs[:,:To,:] = nobs[:,:To,:]

        # (B,T,Do)
        enc_obs = self.obs_encoding_net(obs)

        # Sample latents from the prior
        latents, offsets = self.state_prior.generate_latents(enc_obs)

        # un-descritize
        naction_pred = self.action_ae.decode_actions(
            latent_action_batch=(latents, offsets)
        )
        # (B,T,Da)

        # un-normalize
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def fit_action_ae(self, input_actions: torch.Tensor):
        self.action_ae.fit_discretizer(input_actions=input_actions)
    
    def get_latents(self, latent_collection_loader):
        training_latents = list()
        with eval_mode(self.action_ae, self.obs_encoding_net, no_grad=True):
            for observations, action, mask in latent_collection_loader:
                obs, act = observations.to(self.device, non_blocking=True), action.to(self.device, non_blocking=True)
                enc_obs = self.obs_encoding_net(obs)
                latent = self.action_ae.encode_into_latent(act, enc_obs)
                reconstructed_action = self.action_ae.decode_actions(
                    latent,
                    enc_obs,
                )
                total_mse_loss += F.mse_loss(act, reconstructed_action, reduction="sum")
                if type(latent) == tuple:
                    # serialize into tensor; assumes last dim is latent dim
                    detached_latents = tuple(x.detach() for x in latent)
                    training_latents.append(torch.cat(detached_latents, dim=-1))
                else:
                    training_latents.append(latent.detach())
        training_latents_tensor = torch.cat(training_latents, dim=0)
        return training_latents_tensor

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.state_prior.get_optimizer(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas))
    
    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # mask out observations after n_obs_steps
        To = self.n_obs_steps
        obs[:,To:,:] = -2 # (normal obs range [-1,1])

        enc_obs = self.obs_encoding_net(obs)
        latent = self.action_ae.encode_into_latent(action, enc_obs)
        _, loss, loss_components = self.state_prior.get_latent_and_loss(
            obs_rep=enc_obs,
            target_latents=latent,
            return_loss_components=True,
        )
        return loss, loss_components
