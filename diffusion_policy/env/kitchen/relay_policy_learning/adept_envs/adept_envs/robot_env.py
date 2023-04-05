"""Base class for robotics environments."""

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

import importlib
import inspect
import os
from typing import Dict, Optional

import numpy as np


from adept_envs import mujoco_env
from adept_envs.base_robot import BaseRobot
from adept_envs.utils.configurable import import_class_from_path
from adept_envs.utils.constants import MODELS_PATH


class RobotEnv(mujoco_env.MujocoEnv):
    """Base environment for all adept robots."""

    # Mapping of robot name to fully qualified class path.
    # e.g. 'robot': 'adept_envs.dclaw.robot.Robot'
    # Subclasses should override this to specify the Robot classes they support.
    ROBOTS = {}

    # Mapping of device path to the calibration file to use. If the device path
    # is not found, the 'default' key is used.
    # This can be overridden by subclasses.
    CALIBRATION_PATHS = {}

    def __init__(self,
                 model_path: str,
                 robot: BaseRobot,
                 frame_skip: int,
                 camera_settings: Optional[Dict] = None):
        """Initializes a robotics environment.

        Args:
            model_path: The path to the model to run. Relative paths will be
              interpreted as relative to the 'adept_models' folder.
            robot: The Robot object to use.
            frame_skip: The number of simulation steps per environment step. On
              hardware this influences the duration of each environment step.
            camera_settings: Settings to initialize the simulation camera. This
              can contain the keys `distance`, `azimuth`, and `elevation`.
        """
        self._robot = robot

        # Initial pose for first step.
        self.desired_pose = np.zeros(self.n_jnt)

        if not model_path.startswith('/'):
            model_path = os.path.abspath(os.path.join(MODELS_PATH, model_path))

        self.remote_viz = None

        try:
            from adept_envs.utils.remote_viz import RemoteViz
            self.remote_viz = RemoteViz(model_path)
        except ImportError:
            pass          


        self._initializing = True
        super(RobotEnv, self).__init__(
            model_path, frame_skip, camera_settings=camera_settings)
        self._initializing = False


    @property
    def robot(self):
        return self._robot

    @property
    def n_jnt(self):
        return self._robot.n_jnt

    @property
    def n_obj(self):
        return self._robot.n_obj

    @property
    def skip(self):
        """Alias for frame_skip. Needed for MJRL."""
        return self.frame_skip

    @property
    def initializing(self):
        return self._initializing

    def close_env(self):
        if self._robot is not None:
            self._robot.close()

    def make_robot(self,
                   n_jnt,
                   n_obj=0,
                   is_hardware=False,
                   device_name=None,
                   legacy=False,
                   **kwargs):
        """Creates a new robot for the environment.

        Args:
            n_jnt: The number of joints in the robot.
            n_obj: The number of object joints in the robot environment.
            is_hardware: Whether to run on hardware or not.
            device_name: The device path for the robot hardware.
            legacy: If true, runs using direct dynamixel communication rather
              than DDS.
            kwargs: See BaseRobot for other parameters.

        Returns:
            A Robot object.
        """
        if not self.ROBOTS:
            raise NotImplementedError('Subclasses must override ROBOTS.')

        if is_hardware and not device_name:
            raise ValueError('Must provide device name if running on hardware.')

        robot_name = 'dds_robot' if not legacy and is_hardware else 'robot'
        if robot_name not in self.ROBOTS:
            raise KeyError("Unsupported robot '{}', available: {}".format(
                robot_name, list(self.ROBOTS.keys())))

        cls = import_class_from_path(self.ROBOTS[robot_name])

        calibration_path = None
        if self.CALIBRATION_PATHS:
            if not device_name:
                calibration_name = 'default'
            elif device_name not in self.CALIBRATION_PATHS:
                print('Device "{}" not in CALIBRATION_PATHS; using default.'
                      .format(device_name))
                calibration_name = 'default'
            else:
                calibration_name = device_name

            calibration_path = self.CALIBRATION_PATHS[calibration_name]
            if not os.path.isfile(calibration_path):
                raise OSError('Could not find calibration file at: {}'.format(
                    calibration_path))

        return cls(
            n_jnt,
            n_obj,
            is_hardware=is_hardware,
            device_name=device_name,
            calibration_path=calibration_path,
            **kwargs)
