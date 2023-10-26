from typing import Dict, Sequence, Union, Optional
import numpy as np
import skimage.transform as st
import pymunk
import pygame
from matplotlib import cm
import cv2
from diffusion_policy.env.pusht.pymunk_override import DrawOptions


def farthest_point_sampling(points: np.ndarray, n_points: int, init_idx: int):
    """
    Naive O(N^2)
    """
    assert(n_points >= 1)
    chosen_points = [points[init_idx]]
    for _ in range(n_points-1):
        cpoints = np.array(chosen_points)
        all_dists = np.linalg.norm(points[:,None,:] - cpoints[None,:,:], axis=-1)
        min_dists = all_dists.min(axis=1)
        next_idx = np.argmax(min_dists)
        next_pt = points[next_idx]
        chosen_points.append(next_pt)
    result = np.array(chosen_points)
    return result


class PymunkKeypointManager:
    def __init__(self, 
            local_keypoint_map: Dict[str, np.ndarray], 
            color_map: Optional[Dict[str, np.ndarray]]=None):
        """
        local_keypoint_map:
            "<attribute_name>": (N,2) floats in object local coordinate
        """
        if color_map is None:
            cmap = cm.get_cmap('tab10')
            color_map = dict()
            for i, key in enumerate(local_keypoint_map.keys()):
                color_map[key] = (np.array(cmap.colors[i]) * 255).astype(np.uint8)

        self.local_keypoint_map = local_keypoint_map
        self.color_map = color_map
    
    @property
    def kwargs(self):
        return {
            'local_keypoint_map': self.local_keypoint_map,
            'color_map': self.color_map
        }

    @classmethod
    def create_from_pusht_env(cls, env, n_block_kps=9, n_agent_kps=3, seed=0, **kwargs):
        rng = np.random.default_rng(seed=seed)
        local_keypoint_map = dict()
        for name in ['block','agent']:
            self = env
            self.space = pymunk.Space()
            if name == 'agent':
                self.agent = obj = self.add_circle((256, 400), 15)
                n_kps = n_agent_kps
            else:
                self.block = obj = self.add_tee((256, 300), 0)
                n_kps = n_block_kps
            
            self.screen = pygame.Surface((512,512))
            self.screen.fill(pygame.Color("white"))
            draw_options = DrawOptions(self.screen)
            self.space.debug_draw(draw_options)
            # pygame.display.flip()
            img = np.uint8(pygame.surfarray.array3d(self.screen).transpose(1, 0, 2))
            obj_mask = (img != np.array([255,255,255],dtype=np.uint8)).any(axis=-1)

            tf_img_obj = cls.get_tf_img_obj(obj)
            xy_img = np.moveaxis(np.array(np.indices((512,512))), 0, -1)[:,:,::-1]
            local_coord_img = tf_img_obj.inverse(xy_img.reshape(-1,2)).reshape(xy_img.shape)
            obj_local_coords = local_coord_img[obj_mask]

            # furthest point sampling
            init_idx = rng.choice(len(obj_local_coords))
            obj_local_kps = farthest_point_sampling(obj_local_coords, n_kps, init_idx)
            small_shift = rng.uniform(0, 1, size=obj_local_kps.shape)
            obj_local_kps += small_shift

            local_keypoint_map[name] = obj_local_kps
        
        return cls(local_keypoint_map=local_keypoint_map, **kwargs)

    @staticmethod
    def get_tf_img(pose: Sequence):
        pos = pose[:2]
        rot = pose[2]
        tf_img_obj = st.AffineTransform(
            translation=pos, rotation=rot)
        return tf_img_obj

    @classmethod
    def get_tf_img_obj(cls, obj: pymunk.Body):
        pose = tuple(obj.position) + (obj.angle,)
        return cls.get_tf_img(pose)

    def get_keypoints_global(self, 
            pose_map: Dict[set, Union[Sequence, pymunk.Body]], 
            is_obj=False):
        kp_map = dict()
        for key, value in pose_map.items():
            if is_obj:
                tf_img_obj = self.get_tf_img_obj(value)
            else:
                tf_img_obj = self.get_tf_img(value)
            kp_local = self.local_keypoint_map[key]
            kp_global = tf_img_obj(kp_local)
            kp_map[key] = kp_global
        return kp_map
    
    def draw_keypoints(self, img, kps_map, radius=1):
        scale = np.array(img.shape[:2]) / np.array([512,512])
        for key, value in kps_map.items():
            color = self.color_map[key].tolist()
            coords = (value * scale).astype(np.int32)
            for coord in coords:
                cv2.circle(img, coord, radius=radius, color=color, thickness=-1)
        return img
    
    def draw_keypoints_pose(self, img, pose_map, is_obj=False, **kwargs):
        kp_map = self.get_keypoints_global(pose_map, is_obj=is_obj)
        return self.draw_keypoints(img, kps_map=kp_map, **kwargs)


def test():
    from diffusion_policy.environment.push_t_env import PushTEnv
    from matplotlib import pyplot as plt
    
    env = PushTEnv(headless=True, obs_state=False, draw_action=False)
    kp_manager = PymunkKeypointManager.create_from_pusht_env(env=env)
    env.reset()
    obj_map = {
        'block': env.block,
        'agent': env.agent
    }

    obs = env.render()
    img = obs.astype(np.uint8)
    kp_manager.draw_keypoints_pose(img=img, pose_map=obj_map, is_obj=True)

    plt.imshow(img)
