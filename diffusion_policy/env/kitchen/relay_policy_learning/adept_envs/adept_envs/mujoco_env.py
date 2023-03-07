"""Base environment for MuJoCo-based environments."""

#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import collections
import os
import time
from typing import Dict, Optional

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from adept_envs.simulation.sim_robot import MujocoSimRobot, RenderMode

DEFAULT_RENDER_SIZE = 480

USE_DM_CONTROL = True


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments."""

    def __init__(self,
                 model_path: str,
                 frame_skip: int,
                 camera_settings: Optional[Dict] = None,
                 use_dm_backend: Optional[bool] = None,
                 ):
        """Initializes a new MuJoCo environment.

        Args:
            model_path: The path to the MuJoCo XML file.
            frame_skip: The number of simulation steps per environment step. On
              hardware this influences the duration of each environment step.
            camera_settings: Settings to initialize the simulation camera. This
              can contain the keys `distance`, `azimuth`, and `elevation`.
            use_dm_backend: A boolean to switch between mujoco-py and dm_control.
        """
        self._seed()
        if not os.path.isfile(model_path):
            raise IOError(
                '[MujocoEnv]: Model path does not exist: {}'.format(model_path))
        self.frame_skip = frame_skip

        self.sim_robot = MujocoSimRobot(
            model_path,
            use_dm_backend=use_dm_backend or USE_DM_CONTROL,
            camera_settings=camera_settings)
        self.sim = self.sim_robot.sim
        self.model = self.sim_robot.model
        self.data = self.sim_robot.data

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.mujoco_render_frames = False

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        assert not done

        bounds = self.model.actuator_ctrlrange.copy()
        act_upper = bounds[:, 1]
        act_lower = bounds[:, 0]

        # Define the action and observation spaces.
        # HACK: MJRL is still using gym 0.9.x so we can't provide a dtype.
        try:
            self.action_space = spaces.Box(
                act_lower, act_upper, dtype=np.float32)
            if isinstance(observation, collections.Mapping):
                self.observation_space = spaces.Dict({
                k: spaces.Box(-np.inf, np.inf, shape=v.shape, dtype=np.float32) for k, v in observation.items()})
            else:
                self.obs_dim = np.sum([o.size for o in observation]) if type(observation) is tuple else observation.size
                self.observation_space = spaces.Box(
                -np.inf, np.inf, observation.shape, dtype=np.float32)

        except TypeError:
            # Fallback case for gym 0.9.x
            self.action_space = spaces.Box(act_lower, act_upper)
            assert not isinstance(observation, collections.Mapping), 'gym 0.9.x does not support dictionary observation.'
            self.obs_dim = np.sum([o.size for o in observation]) if type(observation) is tuple else observation.size
            self.observation_space = spaces.Box(
                -np.inf, np.inf, observation.shape)

    def seed(self, seed=None):  # Compatibility with new gym
        return self._seed(seed)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """Reset the robot degrees of freedom (qpos and qvel).

        Implement this in each subclass.
        """
        raise NotImplementedError

    # -----------------------------

    def reset(self):  # compatibility with new gym
        return self._reset()

    def _reset(self):
        self.sim.reset()
        self.sim.forward()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        # we are directly manipulating mujoco state here
        data = self.sim.data # MjData
        for i in range(self.model.nq):
            data.qpos[i] = qpos[i]
        for i in range(self.model.nv):
            data.qvel[i] = qvel[i]
        # state = np.concatenate([self.data.qpos, self.data.qvel, self.data.act])
        # self.sim.set_state(state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        for i in range(self.model.nu):
            self.sim.data.ctrl[i] = ctrl[i]

        for _ in range(n_frames):
            self.sim.step()

            # TODO(michaelahn): Remove this; render should be called separately.
            if self.mujoco_render_frames is True:
                self.mj_render()

    def render(self,
               mode='human',
               width=DEFAULT_RENDER_SIZE,
               height=DEFAULT_RENDER_SIZE,
               camera_id=-1):
        """Renders the environment.

        Args:
            mode: The type of rendering to use.
                - 'human': Renders to a graphical window.
                - 'rgb_array': Returns the RGB image as an np.ndarray.
                - 'depth_array': Returns the depth image as an np.ndarray.
            width: The width of the rendered image. This only affects offscreen
                rendering.
            height: The height of the rendered image. This only affects
                offscreen rendering.
            camera_id: The ID of the camera to use. By default, this is the free
                camera. If specified, only affects offscreen rendering.
        """
        if mode == 'human':
            self.sim_robot.renderer.render_to_window()
        elif mode == 'rgb_array':
            assert width and height
            return self.sim_robot.renderer.render_offscreen(
                width, height, mode=RenderMode.RGB, camera_id=camera_id)
        elif mode == 'depth_array':
            assert width and height
            return self.sim_robot.renderer.render_offscreen(
                width, height, mode=RenderMode.DEPTH, camera_id=camera_id)
        else:
            raise NotImplementedError(mode)

    def close(self):
        self.sim_robot.close()

    def mj_render(self):
        """Backwards compatibility with MJRL."""
        self.render(mode='human')

    def state_vector(self):
        state = self.sim.get_state()
        return np.concatenate([state.qpos.flat, state.qvel.flat])