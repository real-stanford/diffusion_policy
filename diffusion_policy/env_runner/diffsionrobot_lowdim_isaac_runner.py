import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

from ase.run import create_rlgpu_env_cfg

class IsaacHumanoidRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)
        
        cfg = torch.load('nominal_cfg.pt')
        env = create_rlgpu_env_cfg(*cfg)

        self.env = env
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    
    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        obs = env.reset()
        past_action = None
        policy.reset()

        pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval IsaacGym", 
            leave=False, mininterval=self.tqdm_interval_sec)
        done = False
        
        # history = 2
        # state_history = torch.zeros((env.num_envs, history, env.num_states), dtype=torch.float32, device=self.device)
        # action_history = torch.zeros((env.num_envs, history, env.num_actions), dtype=torch.float32, device=self.device)
        
        while not done:
            # create obs dict
            np_obs_dict = {'state': obs}
            if self.past_action and (past_action is not None):
                # TODO: not tested
                np_obs_dict['past_action'] = past_action[
                    :,-(self.n_obs_steps-1):].astype(np.float32)
            
            # device transfer
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).to(
                    device=device))

            # run policy
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())

            action = np_action_dict['action']

            # step env
            obs, reward, done, info = env.step(action)
            
            # reset env
            env_ids = torch.nonzero(done, as_tuple=False).squeeze(1).int()
            obs = env.reset(env_ids)
            
            done = np.all(done)
            past_action = action

            # update pbar
            pbar.update(action.shape[1])
        # clear out video buffer
        _ = env.reset()

        # log

        return None

