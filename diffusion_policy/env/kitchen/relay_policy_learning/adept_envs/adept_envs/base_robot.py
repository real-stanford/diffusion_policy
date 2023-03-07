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

import numpy as np
from collections import deque

class BaseRobot(object):
    """Base class for all robot classes."""

    def __init__(self,
                 n_jnt,
                 n_obj,
                 pos_bounds=None,
                 vel_bounds=None,
                 calibration_path=None,
                 is_hardware=False,
                 device_name=None,
                 overlay=False,
                 calibration_mode=False,
                 observation_cache_maxsize=5):
        """Create a new robot.
        Args:
            n_jnt: The number of dofs in the robot.
            n_obj: The number of dofs in the object.
            pos_bounds: (n_jnt, 2)-shape matrix denoting the min and max joint
                position for each joint.
            vel_bounds: (n_jnt, 2)-shape matrix denoting the min and max joint
                velocity for each joint.
            calibration_path: File path to the calibration configuration file to
                use.
            is_hardware: Whether to run on hardware or not.
            device_name: The device path for the robot hardware. Only required
                in legacy mode.
            overlay: Whether to show a simulation overlay of the hardware.
            calibration_mode: Start with motors disengaged.
        """

        assert n_jnt > 0
        assert n_obj >= 0

        self._n_jnt = n_jnt
        self._n_obj = n_obj
        self._n_dofs = n_jnt + n_obj

        self._pos_bounds = None
        if pos_bounds is not None:
            pos_bounds = np.array(pos_bounds, dtype=np.float32)
            assert pos_bounds.shape == (self._n_dofs, 2)
            for low, high in pos_bounds:
                assert low < high
            self._pos_bounds = pos_bounds
        self._vel_bounds = None
        if vel_bounds is not None:
            vel_bounds = np.array(vel_bounds, dtype=np.float32)
            assert vel_bounds.shape == (self._n_dofs, 2)
            for low, high in vel_bounds:
                assert low < high
            self._vel_bounds = vel_bounds

        self._is_hardware = is_hardware
        self._device_name = device_name
        self._calibration_path = calibration_path
        self._overlay = overlay
        self._calibration_mode = calibration_mode
        self._observation_cache_maxsize = observation_cache_maxsize

        # Gets updated
        self._observation_cache = deque([], maxlen=self._observation_cache_maxsize)


    @property
    def n_jnt(self):
        return self._n_jnt

    @property
    def n_obj(self):
        return self._n_obj

    @property
    def n_dofs(self):
        return self._n_dofs

    @property
    def pos_bounds(self):
        return self._pos_bounds

    @property
    def vel_bounds(self):
        return self._vel_bounds

    @property
    def is_hardware(self):
        return self._is_hardware

    @property
    def device_name(self):
        return self._device_name

    @property
    def calibration_path(self):
        return self._calibration_path

    @property
    def overlay(self):
        return self._overlay

    @property
    def has_obj(self):
        return self._n_obj > 0

    @property
    def calibration_mode(self):
        return self._calibration_mode

    @property
    def observation_cache_maxsize(self):
        return self._observation_cache_maxsize

    @property
    def observation_cache(self):
        return self._observation_cache


    def clip_positions(self, positions):
        """Clips the given joint positions to the position bounds.

        Args:
            positions: The joint positions.

        Returns:
            The bounded joint positions.
        """
        if self.pos_bounds is None:
            return positions
        assert len(positions) == self.n_jnt or len(positions) == self.n_dofs
        pos_bounds = self.pos_bounds[:len(positions)]
        return np.clip(positions, pos_bounds[:, 0], pos_bounds[:, 1])

