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

import zarr, time

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
        
        self.save_zarr = True
        
        if self.save_zarr:
            cfg['env']['numEnvs']=1
            args.num_envs=1
            args.rl_device = 'cpu'
            args.device = 'cpu'
            args.use_gpu_pipeline = False
            args.use_gpu = False
        else: # placeholder
            cfg['env']['numEnvs']=1
            args.num_envs=1
            args.rl_device = 'cpu'
            args.device = 'cpu'
            args.use_gpu_pipeline = False
            args.use_gpu = False
                
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
    
    def run(self, policy: BaseLowdimPolicy, online=False, generate_data=False):
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
        state_history = torch.zeros((env.num_envs, history+1, env.num_obs + 64), dtype=torch.float32, device=device)
        action_history = torch.zeros((env.num_envs, history, env.num_actions), dtype=torch.float32, device=device)
        
        # state_history[:,:,:] = obs[:,None,:]
        state_history[:,:,:] = torch.cat([self.player._ase_latents.to(device), obs.to(device)], dim=-1) [:,None,:] # (env.num_envs, 1, env.num_observations)
        
        obs_dict = {'obs': state_history[:,:]} #, 'past_action': action_history}
        single_obs_dict = {'obs': state_history[:,-1, -253:].to('cuda:0')} #, 'past_action': action_history[0]}
        
        
        save_zarr = generate_data or (not online)
        len_to_save = 1200 if not generate_data else 1e7
        print("length to save", len_to_save)
        if save_zarr:
            
            if generate_data:
                zroot = zarr.open_group("recorded_data{}.zarr".format(time.strftime("%H-%M-%S", time.localtime())), "w")
            else:
                zroot = zarr.open_group("recorded_data_eval.zarr", "w")
            
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
            
            recorded_obs_episode = np.zeros((env.num_envs, env.task.max_episode_length, env.num_observations))
            recorded_acs_episode = np.zeros((env.num_envs, env.task.max_episode_length, env.num_actions))
            recorded_latent_episode = np.zeros((env.num_envs, env.task.max_episode_length, 64))
            
            
        episode_ends = []
        action_error = []
        idx = 0    
        saved_idx = 0    
        skip = 5
        while not done:
            # run policy
            with torch.no_grad():
                expert_action = self.player.get_action(single_obs_dict, is_determenistic=True)
                # if idx % skip == 4: #not save_zarr and 
                if online:
                    
                    
                    # list in isaac: [3,12,16,18,17,19,21,1,4,7,2,5,8]
                    idx_isaac = [3,12,17,19,21,16,18,2,5,8,1,4,7]
                    # list of dof: [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 35, 36, 37]
                    idx_dof = [0, 1,  2,  3,  4,  5,  6,  7,  8,  9, 12, 13, 14, 15, 16, 17, 18,     21, 22, 23, 25, 27, 28,29, 30, 31, 32, 34, 36, 37, 38]
                    # Your theta values: a numpy array of shape (192, 24, 3)
                    thetas = np.load('thetas.npy', allow_pickle=True)  # Load your thetas
                    thetas = torch.tensor(thetas, dtype=torch.float32)
                    thetas[0,:,[16,17]] += thetas[0,:,[13,14]]
                    thetas = thetas[0,:,idx_isaac]
                    old_thetas = torch.clone(thetas)
                    thetas = torch.clone(thetas)
                    thetas[...,0] = old_thetas[...,1]
                    thetas[...,1] = old_thetas[...,0]
                    thetas[...,2] = old_thetas[...,2]
                    thetas[:,[2,5],0] *= -1
                    thetas[:,[2,5],2] *= -1
                    # thetas[:,[3,6],1] *= -1
                    thetas = thetas.view(thetas.shape[0], -1)  
                    thetas = thetas[:,idx_dof] * torch.pi / 180.0
                    
                    
                    
                # if idx > 10:
                    obs_dict = {'obs': state_history[:,-4:,:]}
                    # obs_dict['obs'][:,-1,:-253] = torch.tensor([ 0.0378,  0.0338,  0.0726, -0.0593, -0.1044,  0.1489, -0.0006, -0.1080,
                    # 0.2047, -0.1283,  0.0318,  0.0232, -0.3094,  0.1823,  0.0721,  0.2027,
                    # -0.0547, -0.2114, -0.0986, -0.3038, -0.0201, -0.0805,  0.1522,  0.0325,
                    # 0.0959, -0.0328, -0.1267,  0.0727,  0.2151,  0.1019, -0.0809,  0.0511,
                    # -0.0550,  0.0733, -0.2014,  0.0836,  0.1340,  0.0580, -0.0762, -0.0574,
                    # 0.1326, -0.0737,  0.1634,  0.0785, -0.1303,  0.0276, -0.0330,  0.1069,
                    # 0.2971,  0.1978,  0.1767,  0.0819,  0.0369,  0.0803, -0.0847,  0.0224,
                    # 0.1209, -0.0013, -0.1292, -0.1193,  0.0442, -0.0701, -0.1243,  0.0971]).cuda()
                    # obs_dict['obs'][:,-1,:-253] = torch.mean(obs_dict['obs'][:,-1,:-253], dim=0)
                    # print(obs_dict['obs'][:,-1,:4])
                    # action_dict = policy.predict_action(obs_dict)
                    
                    # use CNN
                    # action_dict = policy.predict_action_half(obs_dict, thetas[idx:idx+policy.horizon])
                    
                    # use TF
                    action_dict = policy.predict_action(obs_dict)
                    
                    if idx >= thetas.shape[0] - policy.horizon:
                        idx = 0
                    # obs_dict_2 = {'obs': torch.cat([
                    #     torch.from_numpy(recorded_latent_episode[0:1]).to('cuda:0'), 
                    #     torch.from_numpy(recorded_obs_episode[0:1]).to('cuda:0')], dim=-1)[:,idx-history:idx,:]}
                    # # print(obs_dict_2['obs'][:,:,-3:], '\n state: \n', obs_dict['obs'][:,:,-3:], '\n curr: \n', single_obs_dict['obs'][:,-3:])
                    # action_dict_2 = policy.predict_action(obs_dict_2)
                    # pred_action = action_dict_2['action_pred']
                    # gt_action = torch.from_numpy(recorded_acs_episode[0:1]).view(1,-1,31).to('cuda:0')[:,idx-history:idx,:]
                    
                    # mse = torch.nn.functional.mse_loss(pred_action[:,:gt_action.shape[1],:], gt_action)
                    # print("\n\n\ngt_action: ", pred_action[:,:gt_action.shape[1],:3], '\n', gt_action[:,:,:3])
                    # print("eval mse: ", mse.item(), np.sqrt(mse.item()))
                    # print('action diff: ', torch.mean(torch.sqrt((expert_action[0] - pred_action[0,history]) ** 2)))
                    # action_error.append(torch.mean(torch.sqrt((expert_action[0] - pred_action[0,history]) ** 2)).item())
                    
                    pred_action = action_dict['action_pred']
                    action = pred_action[:,history:history+3,:]
                    # action = expert_action[:,None,:]
                else:
                    action = expert_action[:,None,:]
            if save_zarr:
                curr_idx = np.all(recorded_latent_episode == 0, axis=-1).argmax(axis=-1)
                # curr_idx = idx
                recorded_obs_episode[np.arange(env.num_envs),curr_idx,:] = single_obs_dict['obs'].to("cpu").detach().numpy()
                recorded_acs_episode[np.arange(env.num_envs),curr_idx+1,:] = expert_action.to("cpu").detach().numpy()
                recorded_latent_episode[np.arange(env.num_envs),curr_idx,:] = self.player._ase_latents.to("cpu").detach().numpy()[:,:]
                
            # print(self.player._ase_latents.to("cpu").detach().numpy()[0,:3])

            # step env
            self.n_action_steps = action.shape[1]
            for i in range(self.n_action_steps):
                action_step = action[:,i,:]
                obs, reward, done, info = env.step(action_step)
            
                state_history = torch.roll(state_history, shifts=-1, dims=1)
                action_history = torch.roll(action_history, shifts=-1, dims=1)
                state_history[:,-1,-253:] = obs
                state_history[:,-1,:-253] = self.player._ase_latents
                
                action_history[:,-1,:] = action_step
                single_obs_dict = {'obs': state_history[:,-1,-253:].to('cuda:0')}
            
                idx += 1
            # reset env
            env_ids = torch.nonzero(done, as_tuple=False).squeeze(1).int()
            # print("env_ids: ", env_ids)
            if len(env_ids) > 0:
                self.player.env_reset(env_ids)
                obs = env.reset(env_ids)
                # state_history[env_ids,:,:] = obs_2[env_ids,None,:]
                state_history[env_ids,:,-253:] = obs[env_ids].to(state_history.device)[:,None,:]
                state_history[env_ids,:,:-253] = self.player._ase_latents[env_ids].to(state_history.device)[:,None,:]
                action_history[env_ids,:,:] = 0.0
                
                idx = 0
                
                # flush saved data
                if save_zarr:
                    for i in range(len(env_ids)):
                        epi_len = np.all(recorded_obs_episode[env_ids[i]] == 0, axis=-1).argmax(axis=-1)+1
                        if epi_len == 0:
                            epi_len = recorded_acs_episode.shape[1]
                        recorded_obs.append(np.copy(recorded_obs_episode[env_ids[i], :epi_len]))
                        recorded_acs.append(np.copy(recorded_acs_episode[env_ids[i], :epi_len]))
                        recorded_latent.append(np.copy(recorded_latent_episode[env_ids[i], :epi_len]))
                        
                        recorded_obs_episode[env_ids[i]] = 0
                        recorded_acs_episode[env_ids[i]] = 0
                        recorded_latent_episode[env_ids[i]] = 0
                        
                        saved_idx += epi_len
                        episode_ends.append(saved_idx)
                    
            done = done.cpu().numpy()
            done = np.all(done)
            past_action = action

            # update pbar
            if online:
                pbar.update(action.shape[1])
            else:
                pbar.update(env.num_envs)
            
            
            if save_zarr and saved_idx >= len_to_save:
                recorded_obs = np.concatenate(recorded_obs)
                recorded_acs = np.concatenate(recorded_acs)
                recorded_latent = np.concatenate(recorded_latent)
                episode_ends = np.array(episode_ends)
                
                zdata["state"] = recorded_obs
                zdata["action"] = recorded_acs
                zdata["ase_latent"] = recorded_latent
                zmeta["episode_ends"] = episode_ends
                print(zroot.tree())
                if generate_data:
                    raise StopIteration
                break
            # elif not save_zarr and idx > 300:
            #     break
            
        # clear out video buffer
        _ = env.reset()
        
        with torch.no_grad():
            batch = {}
            dataset = zarr.open("recorded_data_eval.zarr", "r")
            # sample trajectory from training set, and evaluate difference
            obs = dataset.data.state
            latents = dataset.data.ase_latent
            actions = dataset.data.action
            episode_indices = np.concatenate([np.array([np.arange(i, i + policy.horizon) for i in range(j*100, j*100+20)]) for j in range(10)])
            episode_indices = episode_indices.flatten()
            obs = obs[episode_indices].reshape(-1, policy.horizon, obs.shape[-1])
            latents = latents[episode_indices].reshape(-1, policy.horizon, latents.shape[-1])
            actions = actions[episode_indices].reshape(-1, policy.horizon, actions.shape[-1])
            
            batch['obs'] = np.concatenate([latents, obs], axis=-1)
            # batch['obs'] = obs
            batch['action'] = actions
            batch = dict_apply(batch, torch.from_numpy)
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            obs_dict = {'obs': batch['obs']}
            gt_action = batch['action']
            
            result = policy.predict_action(obs_dict)

            pred_action = result['action_pred']
            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
            
            print("eval mse: ", mse.item(), np.sqrt(mse.item()))
            # release RAM
            del batch
            del obs_dict
            del gt_action
            del result
            del pred_action
            del mse


        # log
        log_data = dict()
        log_data['eval_action_error'] = torch.mean(torch.tensor(action_error))
        print("eval_action_error: ", log_data['eval_action_error'])

        return log_data

