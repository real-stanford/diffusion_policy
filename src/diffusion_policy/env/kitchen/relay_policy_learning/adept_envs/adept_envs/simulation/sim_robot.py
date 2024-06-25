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

"""Module for loading MuJoCo models."""

import os
from typing import Dict, Optional

from adept_envs.simulation import module
from adept_envs.simulation.renderer import DMRenderer, MjPyRenderer, RenderMode


class MujocoSimRobot:
    """Class that encapsulates a MuJoCo simulation.

    This class exposes methods that are agnostic to the simulation backend.
    Two backends are supported:
    1. mujoco_py - MuJoCo v1.50
    2. dm_control - MuJoCo v2.00
    """

    def __init__(self,
                 model_file: str,
                 use_dm_backend: bool = False,
                 camera_settings: Optional[Dict] = None):
        """Initializes a new simulation.

        Args:
            model_file: The MuJoCo XML model file to load.
            use_dm_backend: If True, uses DM Control's Physics (MuJoCo v2.0) as
              the backend for the simulation. Otherwise, uses mujoco_py (MuJoCo
              v1.5) as the backend.
            camera_settings: Settings to initialize the renderer's camera. This
              can contain the keys `distance`, `azimuth`, and `elevation`.
        """
        self._use_dm_backend = use_dm_backend

        if not os.path.isfile(model_file):
            raise ValueError(
                '[MujocoSimRobot] Invalid model file path: {}'.format(
                    model_file))

        if self._use_dm_backend:
            dm_mujoco = module.get_dm_mujoco()
            if model_file.endswith('.mjb'):
                self.sim = dm_mujoco.Physics.from_binary_path(model_file)
            else:
                self.sim = dm_mujoco.Physics.from_xml_path(model_file)
            self.model = self.sim.model
            self._patch_mjlib_accessors(self.model, self.sim.data)
            self.renderer = DMRenderer(
                self.sim, camera_settings=camera_settings)
        else:  # Use mujoco_py
            mujoco_py = module.get_mujoco_py()
            self.model = mujoco_py.load_model_from_path(model_file)
            self.sim = mujoco_py.MjSim(self.model)
            self.renderer = MjPyRenderer(
                self.sim, camera_settings=camera_settings)

        self.data = self.sim.data

    def close(self):
        """Cleans up any resources being used by the simulation."""
        self.renderer.close()

    def save_binary(self, path: str):
        """Saves the loaded model to a binary .mjb file."""
        if os.path.exists(path):
            raise ValueError(
                '[MujocoSimRobot] Path already exists: {}'.format(path))
        if not path.endswith('.mjb'):
            path = path + '.mjb'
        if self._use_dm_backend:
            self.model.save_binary(path)
        else:
            with open(path, 'wb') as f:
                f.write(self.model.get_mjb())

    def get_mjlib(self):
        """Returns an object that exposes the low-level MuJoCo API."""
        if self._use_dm_backend:
            return module.get_dm_mujoco().wrapper.mjbindings.mjlib
        else:
            return module.get_mujoco_py_mjlib()

    def _patch_mjlib_accessors(self, model, data):
        """Adds accessors to the DM Control objects to support mujoco_py API."""
        assert self._use_dm_backend
        mjlib = self.get_mjlib()

        def name2id(type_name, name):
            obj_id = mjlib.mj_name2id(model.ptr,
                                      mjlib.mju_str2Type(type_name.encode()),
                                      name.encode())
            if obj_id < 0:
                raise ValueError('No {} with name "{}" exists.'.format(
                    type_name, name))
            return obj_id

        if not hasattr(model, 'body_name2id'):
            model.body_name2id = lambda name: name2id('body', name)

        if not hasattr(model, 'geom_name2id'):
            model.geom_name2id = lambda name: name2id('geom', name)

        if not hasattr(model, 'site_name2id'):
            model.site_name2id = lambda name: name2id('site', name)

        if not hasattr(model, 'joint_name2id'):
            model.joint_name2id = lambda name: name2id('joint', name)

        if not hasattr(model, 'actuator_name2id'):
            model.actuator_name2id = lambda name: name2id('actuator', name)

        if not hasattr(model, 'camera_name2id'):
            model.camera_name2id = lambda name: name2id('camera', name)

        if not hasattr(data, 'body_xpos'):
            data.body_xpos = data.xpos

        if not hasattr(data, 'body_xquat'):
            data.body_xquat = data.xquat
