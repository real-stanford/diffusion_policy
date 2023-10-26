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

import zarr

from ase import run
from ase.run import create_rlgpu_env_cfg, build_alg_runner, RLGPUAlgoObserver

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
            n_envs=None,
            device=None,
        ):
        super().__init__(output_dir)
        
        args, cfg, cfg_train = torch.load('nominal_cfg.pt')

        # cfg['args'].rl_device = device
        # cfg['args'].sim_device = device
        # cfg['args'].graphics_device_id = int(device[-1])
        # cfg['args'].compute_device_id = int(device[-1])
        # cfg['args'].device_id = int(device[-1])
        
        # args.rl_device = device
        # args.sim_device = device
        # args.device_id = int(device[-1])
                
        run.args = args
        run.cfg = cfg
        run.cfg_train = cfg_train
        
        algo_observer = RLGPUAlgoObserver()
        runner = build_alg_runner(algo_observer)
        runner.load(cfg_train)
        runner.reset()
        player = runner.run(vars(args))

        self.player = player
        self.env = player.env
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
        
        history = self.n_obs_steps
        state_history = torch.zeros((env.num_envs, history, env.num_obs + 64), dtype=torch.float32, device=device)
        action_history = torch.zeros((env.num_envs, history, env.num_actions), dtype=torch.float32, device=device)
        
        # state_history[:,:,:] = obs[:,None,:]
        state_history[:,:,:] = torch.cat([obs, self.player._ase_latents], dim=-1) [:,None,:] # (env.num_envs, 1, env.num_states)
        
        obs_dict = {'obs': state_history} #, 'past_action': action_history}
        single_obs_dict = {'obs': state_history[:,-1, :253].to('cuda:0')} #, 'past_action': action_history[0]}
        
        save_zarr = False
        if save_zarr:
            import time
            zroot = zarr.open_group("recorded_data{}.zarr".format(time.strftime("%H-%M-%S", time.localtime())), "w")
            
            zroot.create_group("data")
            zdata = zroot["data"]
            
            zroot.create_group("meta")
            zmeta = zroot["meta"]
            
            zmeta.create_group("episode_ends")
            
            zdata.create_group("action")
            zdata.create_group("state")
            zdata.create_group("ase_latent")
            
        recorded_obs = []
        recorded_acs = []
        recorded_latent = []
        episode_ends = []
        action_error = []
        idx = 0        
        skip = 4
                
        while not done:
            # run policy
            with torch.no_grad():
                
                expert_action = self.player.get_action(single_obs_dict, is_determenistic=True)
                if not save_zarr and idx % skip == 0:
                    action_dict = policy.predict_action(obs_dict)
                    action = action_dict['action']
                    action_error.append(torch.mean(torch.abs(expert_action[:,].to(device) - action[:,0])))
                
            if save_zarr:
                recorded_obs.append(single_obs_dict['obs'].to("cpu").detach().numpy()[0, :])
                recorded_acs.append(expert_action.to("cpu").detach().numpy()[0, :])
                recorded_latent.append(self.player._ase_latents.to("cpu").detach().numpy()[0, :])

            if save_zarr or True:
                action = expert_action[:,None,:]

            # step env
            self.n_action_steps = action.shape[1]
            for i in range(self.n_action_steps):
                action_step = action[:,i,:]
                obs, reward, done, info = env.step(action_step)
            
                state_history = torch.roll(state_history, shifts=-1, dims=1)
                action_history = torch.roll(action_history, shifts=-1, dims=1)
                state_history[:,-1,:253] = obs
                state_history[:,-1,253:] = self.player._ase_latents
                action_history[:,-1,:] = action_step
                single_obs_dict = {'obs': state_history[:,-1,:253].to('cuda:0')}
                
                if save_zarr:
                    if done[0]:
                        episode_ends.append(idx+1)
                idx += 1
            # reset env
            env_ids = torch.nonzero(done, as_tuple=False).squeeze(1).int()
            if len(env_ids) > 0:
                obs = env.reset(env_ids)
                # state_history[env_ids,:,:] = obs_2[env_ids,None,:]
                state_history[env_ids,:,:253] = obs[env_ids].to(state_history.device)[:,None,:]
                state_history[env_ids,:,253:] = self.player._ase_latents[env_ids].to(state_history.device)[:,None,:]
                action_history[env_ids,:,:] = 0.0
            done = done.cpu().numpy()
            done = np.all(done)
            past_action = action

            # update pbar
            pbar.update(action.shape[1])
            
            if save_zarr and idx > 1e6:
                recorded_obs = np.array(recorded_obs)
                recorded_acs = np.array(recorded_acs)
                recorded_latent = np.array(recorded_latent)
                episode_ends = np.array(episode_ends)
                
                zdata["state"] = recorded_obs
                zdata["action"] = recorded_acs
                zdata["ase_latent"] = recorded_latent
                zmeta["episode_ends"] = episode_ends
                print(zroot.tree())
                raise StopIteration
            elif idx > 300:
                break
            
        # clear out video buffer
        _ = env.reset()

        # log
        log_data = dict()
        log_data['eval_action_error'] = torch.mean(torch.tensor(action_error))
        print("eval_action_error: ", log_data['eval_action_error'])

        return log_data

