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

import os, getpass
import numpy as np
from termcolor import cprint
import time
import copy
import click

from adept_envs import base_robot
from adept_envs.utils.config import (get_config_root_node, read_config_from_node)

# observations structure
from collections import namedtuple
observation = namedtuple('observation', ['time', 'qpos_robot', 'qvel_robot', 'qpos_object', 'qvel_object'])



franka_interface = ''

class Robot(base_robot.BaseRobot):

    """
    Abstracts away the differences between the robot_simulation and robot_hardware

    """

    def __init__(self, *args, **kwargs):
        super(Robot, self).__init__(*args, **kwargs)
        global franka_interface

        # Read robot configurations
        self._read_specs_from_config(robot_configs=self.calibration_path)


        # Robot: Handware
        if self.is_hardware:
            if franka_interface == '':
                raise NotImplementedError()
                from handware.franka import franka

                # initialize franka
                self.franka_interface = franka()
                franka_interface = self.franka_interface
                cprint("Initializing %s Hardware (Status:%d)" % (self.robot_name, self.franka.okay(self.robot_hardware_dof)), 'white', 'on_grey')
            else:
                self.franka_interface = franka_interface
                cprint("Reusing previours Franka session", 'white', 'on_grey')

        # Robot: Simulation
        else:
            self.robot_name = "Franka"
            cprint("Initializing %s sim" % self.robot_name, 'white', 'on_grey')

        # Robot's time
        self.time_start = time.time()
        self.time = time.time()-self.time_start
        self.time_render = -1 # time of rendering


    # read specs from the calibration file
    def _read_specs_from_config(self, robot_configs):
        root, root_name = get_config_root_node(config_file_name=robot_configs)
        self.robot_name = root_name[0]
        self.robot_mode = np.zeros(self.n_dofs, dtype=int)
        self.robot_mj_dof = np.zeros(self.n_dofs, dtype=int)
        self.robot_hardware_dof = np.zeros(self.n_dofs, dtype=int)
        self.robot_scale = np.zeros(self.n_dofs, dtype=float)
        self.robot_offset = np.zeros(self.n_dofs, dtype=float)
        self.robot_pos_bound = np.zeros([self.n_dofs, 2], dtype=float)
        self.robot_vel_bound = np.zeros([self.n_dofs, 2], dtype=float)
        self.robot_pos_noise_amp = np.zeros(self.n_dofs, dtype=float)
        self.robot_vel_noise_amp = np.zeros(self.n_dofs, dtype=float)

        print("Reading configurations for %s" % self.robot_name)
        for i in range(self.n_dofs):
            self.robot_mode[i] = read_config_from_node(root, "qpos"+str(i), "mode", int)
            self.robot_mj_dof[i] = read_config_from_node(root, "qpos"+str(i), "mj_dof", int)
            self.robot_hardware_dof[i] = read_config_from_node(root, "qpos"+str(i), "hardware_dof", int)
            self.robot_scale[i] = read_config_from_node(root, "qpos"+str(i), "scale", float)
            self.robot_offset[i] = read_config_from_node(root, "qpos"+str(i), "offset", float)
            self.robot_pos_bound[i] = read_config_from_node(root, "qpos"+str(i), "pos_bound", float)
            self.robot_vel_bound[i] = read_config_from_node(root, "qpos"+str(i), "vel_bound", float)
            self.robot_pos_noise_amp[i] = read_config_from_node(root, "qpos"+str(i), "pos_noise_amp", float)
            self.robot_vel_noise_amp[i] = read_config_from_node(root, "qpos"+str(i), "vel_noise_amp", float)


    # convert to hardware space
    def _de_calib(self, qp_mj, qv_mj=None):
        qp_ad = (qp_mj-self.robot_offset)/self.robot_scale
        if qv_mj is not None:
            qv_ad = qv_mj/self.robot_scale
            return qp_ad, qv_ad
        else:
            return qp_ad

    # convert to mujoco space
    def _calib(self, qp_ad, qv_ad):
        qp_mj  =  qp_ad* self.robot_scale + self.robot_offset
        qv_mj  =  qv_ad* self.robot_scale
        return qp_mj, qv_mj


    # refresh the observation cache
    def _observation_cache_refresh(self, env):
        for _ in range(self.observation_cache_maxsize):
            self.get_obs(env, sim_mimic_hardware=False)

    # get past observation
    def get_obs_from_cache(self, env, index=-1):
        assert (index>=0 and index<self.observation_cache_maxsize) or \
                (index<0 and index>=-self.observation_cache_maxsize), \
                "cache index out of bound. (cache size is %2d)"%self.observation_cache_maxsize
        obs = self.observation_cache[index]
        if self.has_obj:
            return obs.time, obs.qpos_robot, obs.qvel_robot, obs.qpos_object, obs.qvel_object
        else:
            return obs.time, obs.qpos_robot, obs.qvel_robot


    # get observation
    def get_obs(self, env, robot_noise_ratio=1, object_noise_ratio=1, sim_mimic_hardware=True):
        if self.is_hardware:
            raise NotImplementedError()

        else:
            #Gather simulated observation
            qp = env.sim.data.qpos[:self.n_jnt].copy()
            qv = env.sim.data.qvel[:self.n_jnt].copy()
            if self.has_obj:
                qp_obj = env.sim.data.qpos[-self.n_obj:].copy()
                qv_obj = env.sim.data.qvel[-self.n_obj:].copy()
            else:
                qp_obj = None
                qv_obj = None
            self.time = env.sim.data.time

            # Simulate observation noise
            if not env.initializing:
                qp += robot_noise_ratio*self.robot_pos_noise_amp[:self.n_jnt]*env.np_random.uniform(low=-1., high=1., size=self.n_jnt)
                qv += robot_noise_ratio*self.robot_vel_noise_amp[:self.n_jnt]*env.np_random.uniform(low=-1., high=1., size=self.n_jnt)
                if self.has_obj:
                    qp_obj += robot_noise_ratio*self.robot_pos_noise_amp[-self.n_obj:]*env.np_random.uniform(low=-1., high=1., size=self.n_obj)
                    qv_obj += robot_noise_ratio*self.robot_vel_noise_amp[-self.n_obj:]*env.np_random.uniform(low=-1., high=1., size=self.n_obj)

        # cache observations
        obs = observation(time=self.time, qpos_robot=qp, qvel_robot=qv, qpos_object=qp_obj, qvel_object=qv_obj)
        self.observation_cache.append(obs)

        if self.has_obj:
            return obs.time, obs.qpos_robot, obs.qvel_robot, obs.qpos_object, obs.qvel_object
        else:
            return obs.time, obs.qpos_robot, obs.qvel_robot


    # enforce position specs.
    def ctrl_position_limits(self, ctrl_position):
        ctrl_feasible_position = np.clip(ctrl_position, self.robot_pos_bound[:self.n_jnt, 0], self.robot_pos_bound[:self.n_jnt, 1])
        return ctrl_feasible_position


    # step the robot env
    def step(self, env, ctrl_desired, step_duration, sim_override=False):

        # Populate observation cache during startup
        if env.initializing:
            self._observation_cache_refresh(env)

        # enforce velocity limits
        ctrl_feasible = self.ctrl_velocity_limits(ctrl_desired, step_duration)

        # enforce position limits
        ctrl_feasible = self.ctrl_position_limits(ctrl_feasible)

        # Send controls to the robot
        if self.is_hardware and (not sim_override):
            raise NotImplementedError()
        else:
            env.do_simulation(ctrl_feasible, int(step_duration/env.sim.model.opt.timestep)) # render is folded in here

        # Update current robot state on the overlay
        if self.overlay:
            env.sim.data.qpos[self.n_jnt:2*self.n_jnt] = env.desired_pose.copy()
            env.sim.forward()

        # synchronize time
        if self.is_hardware:
            time_now = (time.time()-self.time_start)
            time_left_in_step = step_duration - (time_now-self.time)
            if(time_left_in_step>0.0001):
                time.sleep(time_left_in_step)
        return 1


    def reset(self, env, reset_pose, reset_vel, overlay_mimic_reset_pose=True, sim_override=False):
        reset_pose = self.clip_positions(reset_pose)

        if self.is_hardware:
            raise NotImplementedError()
        else:
            env.sim.reset()
            env.sim.data.qpos[:self.n_jnt] = reset_pose[:self.n_jnt].copy()
            env.sim.data.qvel[:self.n_jnt] = reset_vel[:self.n_jnt].copy()
            if self.has_obj:
                env.sim.data.qpos[-self.n_obj:] = reset_pose[-self.n_obj:].copy()
                env.sim.data.qvel[-self.n_obj:] = reset_vel[-self.n_obj:].copy()
            env.sim.forward()

        if self.overlay:
            env.sim.data.qpos[self.n_jnt:2*self.n_jnt] = env.desired_pose[:self.n_jnt].copy()
            env.sim.forward()

        # refresh observation cache before exit
        self._observation_cache_refresh(env)


    def close(self):
        if self.is_hardware:
            cprint("Closing Franka hardware... ", 'white', 'on_grey', end='', flush=True)
            status = 0
            raise NotImplementedError()
            cprint("Closed (Status: {})".format(status), 'white', 'on_grey', flush=True)
        else:
            cprint("Closing Franka sim", 'white', 'on_grey', flush=True)


class Robot_PosAct(Robot):

    # enforce velocity sepcs.
    # ALERT: This depends on previous observation. This is not ideal as it breaks MDP addumptions. Be careful
    def ctrl_velocity_limits(self, ctrl_position, step_duration):
        last_obs = self.observation_cache[-1]
        ctrl_desired_vel = (ctrl_position-last_obs.qpos_robot[:self.n_jnt])/step_duration

        ctrl_feasible_vel = np.clip(ctrl_desired_vel, self.robot_vel_bound[:self.n_jnt, 0], self.robot_vel_bound[:self.n_jnt, 1])
        ctrl_feasible_position = last_obs.qpos_robot[:self.n_jnt] + ctrl_feasible_vel*step_duration
        return ctrl_feasible_position


class Robot_VelAct(Robot):

    # enforce velocity sepcs.
    # ALERT: This depends on previous observation. This is not ideal as it breaks MDP addumptions. Be careful
    def ctrl_velocity_limits(self, ctrl_velocity, step_duration):
        last_obs = self.observation_cache[-1]

        ctrl_feasible_vel = np.clip(ctrl_velocity, self.robot_vel_bound[:self.n_jnt, 0], self.robot_vel_bound[:self.n_jnt, 1])
        ctrl_feasible_position = last_obs.qpos_robot[:self.n_jnt] + ctrl_feasible_vel*step_duration
        return ctrl_feasible_position

