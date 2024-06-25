from typing import Dict
import torch
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.common.robomimic_config_util import get_robomimic_config

class RobomimicLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            action_dim, 
            obs_dim,
            algo_name='bc_rnn',
            obs_type='low_dim',
            task_name='square',
            dataset_type='ph',
        ):
        super().__init__()
        # key for robomimic obs input
        # previously this is 'object', 'robot0_eef_pos' etc
        obs_key = 'obs'

        config = get_robomimic_config(
            algo_name=algo_name,
            hdf5_type=obs_type,
            task_name=task_name,
            dataset_type=dataset_type)
        with config.unlocked():
            config.observation.modalities.obs.low_dim = [obs_key]
        
        ObsUtils.initialize_obs_utils_with_config(config)
        model: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes={obs_key: [obs_dim]},
                ac_dim=action_dim,
                device='cpu',
            )
        self.model = model
        self.nets = model.nets
        self.normalizer = LinearNormalizer()
        self.obs_key = obs_key
        self.config = config

    def to(self,*args,**kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.model.device = device
        super().to(*args,**kwargs)
    
    # =========== inference =============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs = self.normalizer['obs'].normalize(obs_dict['obs'])
        assert obs.shape[1] == 1
        robomimic_obs_dict = {self.obs_key: obs[:,0,:]}
        naction = self.model.get_action(robomimic_obs_dict)
        action = self.normalizer['action'].unnormalize(naction)
        # (B, Da)
        result = {
            'action': action[:,None,:] # (B, 1, Da)
        }
        return result
    
    def reset(self):
        self.model.reset()
        
    # =========== training ==============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def train_on_batch(self, batch, epoch, validate=False):
        nbatch = self.normalizer.normalize(batch)
        robomimic_batch = {
            'obs': {self.obs_key: nbatch['obs']},
            'actions': nbatch['action']
        }
        input_batch = self.model.process_batch_for_training(
            robomimic_batch)
        info = self.model.train_on_batch(
            batch=input_batch, epoch=epoch, validate=validate)
        # keys: losses, predictions
        return info

    def get_optimizer(self):
        return self.model.optimizers['policy']
