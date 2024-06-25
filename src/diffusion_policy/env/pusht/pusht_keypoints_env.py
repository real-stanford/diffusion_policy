from typing import Dict, Sequence, Union, Optional
from gym import spaces
from diffusion_policy.env.pusht.pusht_env import PushTEnv
from diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
import numpy as np

class PushTKeypointsEnv(PushTEnv):
    def __init__(self,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96,
            keypoint_visible_rate=1.0, 
            agent_keypoints=False,
            draw_keypoints=False,
            reset_to_state=None,
            render_action=True,
            local_keypoint_map: Dict[str, np.ndarray]=None, 
            color_map: Optional[Dict[str, np.ndarray]]=None):
        super().__init__(
            legacy=legacy, 
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            reset_to_state=reset_to_state,
            render_action=render_action)
        ws = self.window_size

        if local_keypoint_map is None:
            # create default keypoint definition
            kp_kwargs = self.genenerate_keypoint_manager_params()
            local_keypoint_map = kp_kwargs['local_keypoint_map']
            color_map = kp_kwargs['color_map']

        # create observation spaces
        Dblockkps = np.prod(local_keypoint_map['block'].shape)
        Dagentkps = np.prod(local_keypoint_map['agent'].shape)
        Dagentpos = 2

        Do = Dblockkps
        if agent_keypoints:
            # blockkp + agnet_pos
            Do += Dagentkps
        else:
            # blockkp + agnet_kp
            Do += Dagentpos
        # obs + obs_mask
        Dobs = Do * 2

        low = np.zeros((Dobs,), dtype=np.float64)
        high = np.full_like(low, ws)
        # mask range 0-1
        high[Do:] = 1.

        # (block_kps+agent_kps, xy+confidence)
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=np.float64
        )

        self.keypoint_visible_rate = keypoint_visible_rate
        self.agent_keypoints = agent_keypoints
        self.draw_keypoints = draw_keypoints
        self.kp_manager = PymunkKeypointManager(
            local_keypoint_map=local_keypoint_map,
            color_map=color_map)
        self.draw_kp_map = None

    @classmethod
    def genenerate_keypoint_manager_params(cls):
        env = PushTEnv()
        kp_manager = PymunkKeypointManager.create_from_pusht_env(env)
        kp_kwargs = kp_manager.kwargs
        return kp_kwargs

    def _get_obs(self):
        # get keypoints
        obj_map = {
            'block': self.block
        }
        if self.agent_keypoints:
            obj_map['agent'] = self.agent

        kp_map = self.kp_manager.get_keypoints_global(
            pose_map=obj_map, is_obj=True)
        # python dict guerentee order of keys and values
        kps = np.concatenate(list(kp_map.values()), axis=0)

        # select keypoints to drop
        n_kps = kps.shape[0]
        visible_kps = self.np_random.random(size=(n_kps,)) < self.keypoint_visible_rate
        kps_mask = np.repeat(visible_kps[:,None], 2, axis=1)

        # save keypoints for rendering
        vis_kps = kps.copy()
        vis_kps[~visible_kps] = 0
        draw_kp_map = {
            'block': vis_kps[:len(kp_map['block'])]
        }
        if self.agent_keypoints:
            draw_kp_map['agent'] = vis_kps[len(kp_map['block']):]
        self.draw_kp_map = draw_kp_map
        
        # construct obs
        obs = kps.flatten()
        obs_mask = kps_mask.flatten()
        if not self.agent_keypoints:
            # passing agent position when keypoints are not available
            agent_pos = np.array(self.agent.position)
            obs = np.concatenate([
                obs, agent_pos
            ])
            obs_mask = np.concatenate([
                obs_mask, np.ones((2,), dtype=bool)
            ])

        # obs, obs_mask
        obs = np.concatenate([
            obs, obs_mask.astype(obs.dtype)
        ], axis=0)
        return obs
    
    
    def _render_frame(self, mode):
        img = super()._render_frame(mode)
        if self.draw_keypoints:
            self.kp_manager.draw_keypoints(
                img, self.draw_kp_map, radius=int(img.shape[0]/96))
        return img
