from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy

class IbcDfoLowdimPolicy(BaseLowdimPolicy):
    def __init__(self,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            dropout=0.1,
            train_n_neg=128,
            pred_n_iter=5,
            pred_n_samples=16384,
            kevin_inference=False,
            andy_train=False
        ):
        super().__init__()

        in_action_channels = action_dim * n_action_steps
        in_obs_channels = obs_dim * n_obs_steps
        in_channels = in_action_channels + in_obs_channels
        mid_channels = 1024
        out_channels = 1

        self.dense0 = nn.Linear(in_features=in_channels, out_features=mid_channels)
        self.drop0 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop2 = nn.Dropout(dropout)
        self.dense3 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop3 = nn.Dropout(dropout)
        self.dense4 = nn.Linear(in_features=mid_channels, out_features=out_channels)

        self.normalizer = LinearNormalizer()

        self.train_n_neg = train_n_neg
        self.pred_n_iter = pred_n_iter
        self.pred_n_samples = pred_n_samples
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.horizon = horizon
        self.kevin_inference = kevin_inference
        self.andy_train = andy_train
    
    def forward(self, obs, action):
        B, N, Ta, Da = action.shape
        B, To, Do = obs.shape
        s = obs.reshape(B,1,-1).expand(-1,N,-1)
        x = torch.cat([s, action.reshape(B,N,-1)], dim=-1).reshape(B*N,-1)
        x = self.drop0(torch.relu(self.dense0(x)))
        x = self.drop1(torch.relu(self.dense1(x)))
        x = self.drop2(torch.relu(self.dense2(x)))
        x = self.drop3(torch.relu(self.dense3(x)))
        x = self.dense4(x)
        x = x.reshape(B,N)
        return x

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
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim
        Ta = self.n_action_steps

        # only take necessary obs
        this_obs = nobs[:,:To]
        naction_stats = self.get_naction_stats()

        # first sample
        action_dist = torch.distributions.Uniform(
            low=naction_stats['min'],
            high=naction_stats['max']
        )
        samples = action_dist.sample((B, self.pred_n_samples, Ta)).to(
            dtype=this_obs.dtype)
        # (B, N, Ta, Da)

        if self.kevin_inference:
            # kevin's implementation
            noise_scale = 3e-2
            for i in range(self.pred_n_iter):
                # Compute energies.
                logits = self.forward(this_obs, samples)
                probs = F.softmax(logits, dim=-1)

                # Resample with replacement.
                idxs = torch.multinomial(probs, self.pred_n_samples, replacement=True)
                samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

                # Add noise and clip to target bounds.
                samples = samples + torch.randn_like(samples) * noise_scale
                samples = samples.clamp(min=naction_stats['min'], max=naction_stats['max'])

            # Return target with highest probability.
            logits = self.forward(this_obs, samples)
            probs = F.softmax(logits, dim=-1)
            best_idxs = probs.argmax(dim=-1)
            acts_n = samples[torch.arange(samples.size(0)), best_idxs, :]
        else:
            # andy's implementation
            zero = torch.tensor(0, device=self.device)
            resample_std = torch.tensor(3e-2, device=self.device)
            for i in range(self.pred_n_iter):
                # Forward pass.
                logits = self.forward(this_obs, samples) # (B, N)
                prob = torch.softmax(logits, dim=-1)

                if i < (self.pred_n_iter - 1):
                    idxs = torch.multinomial(prob, self.pred_n_samples, replacement=True)
                    samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]
                    samples += torch.normal(zero, resample_std, size=samples.shape, device=self.device)

            # Return one sample per x in batch.
            idxs = torch.multinomial(prob, num_samples=1, replacement=True)
            acts_n = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs].squeeze(1)

        action = self.normalizer['action'].unnormalize(acts_n)
        result = {
            'action': action
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        nobs = nbatch['obs']
        naction = nbatch['action']

        # shapes
        Do = self.obs_dim
        Da = self.action_dim
        To = self.n_obs_steps
        Ta = self.n_action_steps
        T = self.horizon
        B = naction.shape[0]

        this_obs = nobs[:,:To]
        start = To - 1
        end = start + Ta
        this_action = naction[:,start:end]

        # Small additive noise to true positives.
        this_action += torch.normal(mean=0, std=1e-4,
            size=this_action.shape,
            dtype=this_action.dtype,
            device=this_action.device)

        # Sample negatives: (B, train_n_neg, Ta, Da)
        naction_stats = self.get_naction_stats()
        action_dist = torch.distributions.Uniform(
            low=naction_stats['min'],
            high=naction_stats['max']
        )
        samples = action_dist.sample((B, self.train_n_neg, Ta)).to(
            dtype=this_action.dtype)
        action_samples = torch.cat([
            this_action.unsqueeze(1), samples], dim=1)
        # (B, train_n_neg+1, Ta, Da)

        if self.andy_train:
            # Get onehot labels
            labels = torch.zeros(action_samples.shape[:2], 
                dtype=this_action.dtype, device=this_action.device)
            labels[:,0] = 1
            logits = self.forward(this_obs, action_samples)
            # (B, N)
            logits = torch.log_softmax(logits, dim=-1)
            loss = -torch.mean(torch.sum(logits * labels, axis=-1))
        else:
            labels = torch.zeros((B,),dtype=torch.int64, device=this_action.device)
            # training
            logits = self.forward(this_obs, action_samples)
            loss = F.cross_entropy(logits, labels)
        return loss


    def get_naction_stats(self):
        Da = self.action_dim
        naction_stats = self.normalizer['action'].get_output_stats()
        repeated_stats = dict()
        for key, value in naction_stats.items():
            assert len(value.shape) == 1
            n_repeats = Da // value.shape[0]
            assert value.shape[0] * n_repeats == Da
            repeated_stats[key] = value.repeat(n_repeats)
        return repeated_stats
