# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multimodal block environments for the XArm."""

import collections
import logging
import math
from typing import Dict, List, Optional, Union
import copy
import time

from gym import spaces
from gym.envs import registration
from diffusion_policy.env.block_pushing import block_pushing
from diffusion_policy.env.block_pushing.utils import utils_pybullet
from diffusion_policy.env.block_pushing.utils.pose3d import Pose3d
from diffusion_policy.env.block_pushing.utils.utils_pybullet import ObjState
from diffusion_policy.env.block_pushing.utils.utils_pybullet import XarmState
import numpy as np
from scipy.spatial import transform
import pybullet
import pybullet_utils.bullet_client as bullet_client

# pytype: skip-file
BLOCK2_URDF_PATH = "third_party/py/envs/assets/block2.urdf"
ZONE2_URDF_PATH = "third_party/py/envs/assets/zone2.urdf"

# When resetting multiple targets, they should all be this far apart.
MIN_BLOCK_DIST = 0.1
MIN_TARGET_DIST = 0.12
# pylint: enable=line-too-long
NUM_RESET_ATTEMPTS = 1000

# Random movement of blocks
RANDOM_X_SHIFT = 0.1
RANDOM_Y_SHIFT = 0.15

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="w",
)
logger = logging.getLogger()


def build_env_name(task, shared_memory, use_image_obs):
    """Construct the env name from parameters."""
    del task
    env_name = "BlockPushMultimodal"

    if use_image_obs:
        env_name = env_name + "Rgb"

    if shared_memory:
        env_name = "Shared" + env_name

    env_name = env_name + "-v0"

    return env_name


class BlockPushEventManager:
    def __init__(self):
        self.event_steps = {
            'REACH_0': -1,
            'REACH_1': -1,
            'TARGET_0_0': -1,
            'TARGET_0_1': -1,
            'TARGET_1_0': -1,
            'TARGET_1_1': -1
        }
    
    def reach(self, step, block_id):
        key = f'REACH_{block_id}'
        if self.event_steps[key] < 0:
            self.event_steps[key] = step
    
    def target(self, step, block_id, target_id):
        key = f'TARGET_{block_id}_{target_id}'
        if self.event_steps[key] < 0:
            self.event_steps[key] = step

    def reset(self):
        for key in list(self.event_steps):
            self.event_steps[key] = -1
    
    def get_info(self):
        return copy.deepcopy(self.event_steps)

class BlockPushMultimodal(block_pushing.BlockPush):
    """2 blocks, 2 targets."""

    def __init__(
        self,
        control_frequency=10.0,
        task=block_pushing.BlockTaskVariant.PUSH,
        image_size=None,
        shared_memory=False,
        seed=None,
        goal_dist_tolerance=0.05,
        abs_action=False
    ):
        """Creates an env instance.

        Args:
          control_frequency: Control frequency for the arm. Each env step will
            advance the simulation by 1/control_frequency seconds.
          task: enum for which task, see BlockTaskVariant enum.
          image_size: Optional image size (height, width). If None, no image
            observations will be used.
          shared_memory: If True `pybullet.SHARED_MEMORY` is used to connect to
            pybullet. Useful to debug.
          seed: Optional seed for the environment.
          goal_dist_tolerance: float, how far away from the goal to terminate.
        """
        self._target_ids = None
        self._target_poses = None
        self._event_manager = BlockPushEventManager()
        super(BlockPushMultimodal, self).__init__(
            control_frequency=control_frequency,
            task=task,
            image_size=image_size,
            shared_memory=shared_memory,
            seed=seed,
            goal_dist_tolerance=goal_dist_tolerance,
        )
        self._init_distance = [-1.0, -1.0]
        self._in_target = [[-1.0, -1.0], [-1.0, -1.0]]
        self._first_move = [-1, -1]
        self._step_num = 0
        self._abs_action = abs_action

    @property
    def target_poses(self):
        return self._target_poses

    def get_goal_translation(self):
        """Return the translation component of the goal (2D)."""
        if self._target_poses:
            return [i.translation for i in self._target_poses]
        else:
            return None

    def _setup_pybullet_scene(self):
        self._pybullet_client = bullet_client.BulletClient(self._connection_mode)

        # Temporarily disable rendering to speed up loading URDFs.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        self._setup_workspace_and_robot()

        self._target_ids = [
            utils_pybullet.load_urdf(self._pybullet_client, i, useFixedBase=True)
            for i in [block_pushing.ZONE_URDF_PATH, ZONE2_URDF_PATH]
        ]
        self._block_ids = []
        for i in [block_pushing.BLOCK_URDF_PATH, BLOCK2_URDF_PATH]:
            self._block_ids.append(
                utils_pybullet.load_urdf(self._pybullet_client, i, useFixedBase=False)
            )

        # Re-enable rendering.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

        self.step_simulation_to_stabilize()

    def _reset_block_poses(self, workspace_center_x):
        """Resets block poses."""

        # Helper for choosing random block position.
        def _reset_block_pose(idx, add=0.0, avoid=None):
            def _get_random_translation():
                block_x = (
                    workspace_center_x
                    + add
                    + self._rng.uniform(low=-RANDOM_X_SHIFT, high=RANDOM_X_SHIFT)
                )
                block_y = -0.2 + self._rng.uniform(
                    low=-RANDOM_Y_SHIFT, high=RANDOM_Y_SHIFT
                )
                block_translation = np.array([block_x, block_y, 0])
                return block_translation

            if avoid is None:
                block_translation = _get_random_translation()
            else:
                # Reject targets too close to `avoid`.
                for _ in range(NUM_RESET_ATTEMPTS):
                    block_translation = _get_random_translation()
                    dist = np.linalg.norm(block_translation[0] - avoid[0])
                    # print('block inner try_idx %d, dist %.3f' % (try_idx, dist))
                    if dist > MIN_BLOCK_DIST:
                        break
            block_sampled_angle = self._rng.uniform(math.pi)
            block_rotation = transform.Rotation.from_rotvec([0, 0, block_sampled_angle])
            self._pybullet_client.resetBasePositionAndOrientation(
                self._block_ids[idx],
                block_translation.tolist(),
                block_rotation.as_quat().tolist(),
            )
            return block_translation

        # Reject targets too close to `avoid`.
        for _ in range(NUM_RESET_ATTEMPTS):
            # Reset first block.
            b0_translation = _reset_block_pose(0)
            # Reset second block away from first block.
            b1_translation = _reset_block_pose(1, avoid=b0_translation)
            dist = np.linalg.norm(b0_translation[0] - b1_translation[0])
            if dist > MIN_BLOCK_DIST:
                break
        else:
            raise ValueError("could not find matching block")
        assert dist > MIN_BLOCK_DIST

    def _reset_target_poses(self, workspace_center_x):
        """Resets target poses."""

        def _reset_target_pose(idx, add=0.0, avoid=None):
            def _get_random_translation():
                # Choose x,y randomly.
                target_x = (
                    workspace_center_x
                    + add
                    + self._rng.uniform(
                        low=-0.05 * RANDOM_X_SHIFT, high=0.05 * RANDOM_X_SHIFT
                    )
                )
                target_y = 0.2 + self._rng.uniform(
                    low=-0.05 * RANDOM_Y_SHIFT, high=0.05 * RANDOM_Y_SHIFT
                )
                target_translation = np.array([target_x, target_y, 0.020])
                return target_translation

            if avoid is None:
                target_translation = _get_random_translation()
            else:
                # Reject targets too close to `avoid`.
                for _ in range(NUM_RESET_ATTEMPTS):
                    target_translation = _get_random_translation()
                    dist = np.linalg.norm(target_translation[0] - avoid[0])
                    # print('target inner try_idx %d, dist %.3f' % (try_idx, dist))
                    if dist > MIN_TARGET_DIST:
                        break
            target_sampled_angle = math.pi + self._rng.uniform(
                low=-math.pi / 30, high=math.pi / 30
            )
            target_rotation = transform.Rotation.from_rotvec(
                [0, 0, target_sampled_angle]
            )
            self._pybullet_client.resetBasePositionAndOrientation(
                self._target_ids[idx],
                target_translation.tolist(),
                target_rotation.as_quat().tolist(),
            )
            self._target_poses[idx] = Pose3d(
                rotation=target_rotation, translation=target_translation
            )

        if self._target_poses is None:
            self._target_poses = [None for _ in range(len(self._target_ids))]

        for _ in range(NUM_RESET_ATTEMPTS):
            # Choose the first target.
            add = 0.12 * self._rng.choice([-1, 1])
            # Randomly flip the location of the targets.
            _reset_target_pose(0, add=add)
            _reset_target_pose(1, add=-add, avoid=self._target_poses[0].translation)
            dist = np.linalg.norm(
                self._target_poses[0].translation[0]
                - self._target_poses[1].translation[0]
            )
            if dist > MIN_TARGET_DIST:
                break
        else:
            raise ValueError("could not find matching target")
        assert dist > MIN_TARGET_DIST

    def _reset_object_poses(self, workspace_center_x, workspace_center_y):
        # Reset block poses.
        self._reset_block_poses(workspace_center_x)

        # Reset target poses.
        self._reset_target_poses(workspace_center_x)

        self._init_distance = [-1.0, -1.0]
        self._in_target = [[-1.0, -1.0], [-1.0, -1.0]]
        self._step_num = 0

    def reset(self, reset_poses=True):
        workspace_center_x = 0.4
        workspace_center_y = 0.0

        if reset_poses:
            self._pybullet_client.restoreState(self._saved_state)

            rotation = transform.Rotation.from_rotvec([0, math.pi, 0])
            translation = np.array([0.3, -0.4, block_pushing.EFFECTOR_HEIGHT])
            starting_pose = Pose3d(rotation=rotation, translation=translation)
            self._set_robot_target_effector_pose(starting_pose)
            self._reset_object_poses(workspace_center_x, workspace_center_y)

        # else:
        self._target_poses = [
            self._get_target_pose(idx) for idx in self._target_ids
        ]

        if reset_poses:
            self.step_simulation_to_stabilize()

        state = self._compute_state()
        self._previous_state = state
        self._event_manager.reset()
        return state

    def _get_target_pose(self, idx):
        (
            target_translation,
            target_orientation_quat,
        ) = self._pybullet_client.getBasePositionAndOrientation(idx)
        target_rotation = transform.Rotation.from_quat(target_orientation_quat)
        target_translation = np.array(target_translation)
        return Pose3d(rotation=target_rotation, translation=target_translation)

    def _compute_reach_target(self, state):
        xy_block = state["block_translation"]
        xy_target = state["target_translation"]

        xy_block_to_target = xy_target - xy_block
        xy_dir_block_to_target = (xy_block_to_target) / np.linalg.norm(
            xy_block_to_target
        )
        self.reach_target_translation = xy_block + -1 * xy_dir_block_to_target * 0.05

    def _compute_state(self):
        effector_pose = self._robot.forward_kinematics()

        def _get_block_pose(idx):
            block_position_and_orientation = (
                self._pybullet_client.getBasePositionAndOrientation(
                    self._block_ids[idx]
                )
            )
            block_pose = Pose3d(
                rotation=transform.Rotation.from_quat(
                    block_position_and_orientation[1]
                ),
                translation=block_position_and_orientation[0],
            )
            return block_pose

        block_poses = [_get_block_pose(i) for i in range(len(self._block_ids))]

        def _yaw_from_pose(pose):
            return np.array([pose.rotation.as_euler("xyz", degrees=False)[-1] % np.pi])

        obs = collections.OrderedDict(
            block_translation=block_poses[0].translation[0:2],
            block_orientation=_yaw_from_pose(block_poses[0]),
            block2_translation=block_poses[1].translation[0:2],
            block2_orientation=_yaw_from_pose(block_poses[1]),
            effector_translation=effector_pose.translation[0:2],
            effector_target_translation=self._target_effector_pose.translation[0:2],
            target_translation=self._target_poses[0].translation[0:2],
            target_orientation=_yaw_from_pose(self._target_poses[0]),
            target2_translation=self._target_poses[1].translation[0:2],
            target2_orientation=_yaw_from_pose(self._target_poses[1]),
        )

        for i in range(2):
            new_distance = np.linalg.norm(
                block_poses[i].translation[0:2]
            )  # + np.linalg.norm(_yaw_from_pose(block_poses[i]))
            if self._init_distance[i] == -1:
                self._init_distance[i] = new_distance
            else:
                if self._init_distance[i] != 100:
                    if np.abs(new_distance - self._init_distance[i]) > 1e-3:
                        logger.info(f"Block {i} moved on step {self._step_num}")
                        self._event_manager.reach(step=self._step_num, block_id=i)
                        self._init_distance[i] = 100

        self._step_num += 1
        if self._image_size is not None:
            obs["rgb"] = self._render_camera(self._image_size)
        return obs

    def step(self, action):
        self._step_robot_and_sim(action)

        state = self._compute_state()
        done = False
        reward = self._get_reward(state)
        if reward >= 0.5:
            # Terminate the episode if both blocks are close enough to the targets.
            done = True

        info = self._event_manager.get_info()
        return state, reward, done, info
    
    def _step_robot_and_sim(self, action):
        """Steps the robot and pybullet sim."""
        # Compute target_effector_pose by shifting the effector's pose by the
        # action.
        if self._abs_action:
            target_effector_translation = np.array([action[0], action[1], 0])
        else:
            target_effector_translation = np.array(
                self._target_effector_pose.translation
            ) + np.array([action[0], action[1], 0])

        target_effector_translation[0:2] = np.clip(
            target_effector_translation[0:2],
            self.workspace_bounds[0],
            self.workspace_bounds[1],
        )
        target_effector_translation[-1] = self.effector_height
        target_effector_pose = Pose3d(
            rotation=block_pushing.EFFECTOR_DOWN_ROTATION, translation=target_effector_translation
        )

        self._set_robot_target_effector_pose(target_effector_pose)

        # Update sleep time dynamically to stay near real-time.
        frame_sleep_time = 0
        if self._connection_mode == pybullet.SHARED_MEMORY:
            cur_time = time.time()
            if self._last_loop_time is not None:
                # Calculate the total, non-sleeping time from the previous frame, this
                # includes the actual step as well as any compute that happens in the
                # caller thread (model inference, etc).
                compute_time = (
                    cur_time
                    - self._last_loop_time
                    - self._last_loop_frame_sleep_time * self._sim_steps_per_step
                )
                # Use this to calculate the current frame's total sleep time to ensure
                # that env.step runs at policy rate. This is an estimate since the
                # previous frame's compute time may not match the current frame.
                total_sleep_time = max((1 / self._control_frequency) - compute_time, 0)
                # Now spread this out over the inner sim steps. This doesn't change
                # control in any way, but makes the animation appear smooth.
                frame_sleep_time = total_sleep_time / self._sim_steps_per_step
            else:
                # No estimate of the previous frame's compute, assume it is zero.
                frame_sleep_time = 1 / self._step_frequency

            # Cache end of this loop time, to compute sleep time on next iteration.
            self._last_loop_time = cur_time
            self._last_loop_frame_sleep_time = frame_sleep_time

        for _ in range(self._sim_steps_per_step):
            if self._connection_mode == pybullet.SHARED_MEMORY:
                block_pushing.sleep_spin(frame_sleep_time)
            self._pybullet_client.stepSimulation()

    def _get_reward(self, state):
        # Reward is 1. if both blocks are inside targets, but not the same target.
        targets = ["target", "target2"]

        def _block_target_dist(block, target):
            return np.linalg.norm(
                state["%s_translation" % block] - state["%s_translation" % target]
            )

        def _closest_target(block):
            # Distances to all targets.
            dists = [_block_target_dist(block, t) for t in targets]
            # Which is closest.
            closest_target = targets[np.argmin(dists)]
            closest_dist = np.min(dists)
            # Is it in the closest target?
            in_target = closest_dist < self.goal_dist_tolerance
            return closest_target, in_target

        blocks = ["block", "block2"]

        reward = 0.0

        for t_i, t in enumerate(targets):
            for b_i, b in enumerate(blocks):
                if self._in_target[t_i][b_i] == -1:
                    dist = _block_target_dist(b, t)
                    if dist < self.goal_dist_tolerance:
                        self._in_target[t_i][b_i] = 0
                        logger.info(
                            f"Block {b_i} entered target {t_i} on step {self._step_num}"
                        )
                        self._event_manager.target(step=self._step_num, block_id=b_i, target_id=t_i)
                        reward += 0.49

        b0_closest_target, b0_in_target = _closest_target("block")
        b1_closest_target, b1_in_target = _closest_target("block2")
        # reward = 0.0
        if b0_in_target and b1_in_target and (b0_closest_target != b1_closest_target):
            reward = 0.51
        return reward

    def _compute_goal_distance(self, state):
        blocks = ["block", "block2"]

        def _target_block_dist(target, block):
            return np.linalg.norm(
                state["%s_translation" % block] - state["%s_translation" % target]
            )

        def _closest_block_dist(target):
            dists = [_target_block_dist(target, b) for b in blocks]
            closest_dist = np.min(dists)
            return closest_dist

        t0_closest_dist = _closest_block_dist("target")
        t1_closest_dist = _closest_block_dist("target2")
        return np.mean([t0_closest_dist, t1_closest_dist])

    @property
    def succeeded(self):
        state = self._compute_state()
        reward = self._get_reward(state)
        if reward >= 0.5:
            return True
        return False

    def _create_observation_space(self, image_size):
        pi2 = math.pi * 2

        obs_dict = collections.OrderedDict(
            block_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            block_orientation=spaces.Box(low=-pi2, high=pi2, shape=(1,)),  # phi
            block2_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            block2_orientation=spaces.Box(low=-pi2, high=pi2, shape=(1,)),  # phi
            effector_translation=spaces.Box(
                low=block_pushing.WORKSPACE_BOUNDS[0] - 0.1,
                high=block_pushing.WORKSPACE_BOUNDS[1] + 0.1,
            ),  # x,y
            effector_target_translation=spaces.Box(
                low=block_pushing.WORKSPACE_BOUNDS[0] - 0.1,
                high=block_pushing.WORKSPACE_BOUNDS[1] + 0.1,
            ),  # x,y
            target_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            target_orientation=spaces.Box(
                low=-pi2,
                high=pi2,
                shape=(1,),
            ),  # theta
            target2_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            target2_orientation=spaces.Box(
                low=-pi2,
                high=pi2,
                shape=(1,),
            ),  # theta
        )
        if image_size is not None:
            obs_dict["rgb"] = spaces.Box(
                low=0, high=255, shape=(image_size[0], image_size[1], 3), dtype=np.uint8
            )
        return spaces.Dict(obs_dict)

    def get_pybullet_state(self):
        """Save pybullet state of the scene.

        Returns:
          dict containing 'robots', 'robot_end_effectors', 'targets', 'objects',
            each containing a list of ObjState.
        """
        state: Dict[str, List[ObjState]] = {}

        state["robots"] = [
            XarmState.get_bullet_state(
                self._pybullet_client,
                self.robot.xarm,
                target_effector_pose=self._target_effector_pose,
                goal_translation=None,
            )
        ]

        state["robot_end_effectors"] = []
        if self.robot.end_effector:
            state["robot_end_effectors"].append(
                ObjState.get_bullet_state(
                    self._pybullet_client, self.robot.end_effector
                )
            )

        state["targets"] = []
        if self._target_ids:
            for target_id in self._target_ids:
                state["targets"].append(
                    ObjState.get_bullet_state(self._pybullet_client, target_id)
                )

        state["objects"] = []
        for obj_id in self.get_obj_ids():
            state["objects"].append(
                ObjState.get_bullet_state(self._pybullet_client, obj_id)
            )

        return state

    def set_pybullet_state(self, state):
        """Restore pyullet state.

        WARNING: py_environment wrapper assumes environments aren't reset in their
        constructor and will often reset the environment unintentionally. It is
        always recommended that you call env.reset on the tfagents wrapper before
        playback (replaying pybullet_state).

        Args:
          state: dict containing 'robots', 'robot_end_effectors', 'targets',
            'objects', each containing a list of ObjState.
        """

        assert isinstance(state["robots"][0], XarmState)
        xarm_state: XarmState = state["robots"][0]
        xarm_state.set_bullet_state(self._pybullet_client, self.robot.xarm)
        self._set_robot_target_effector_pose(xarm_state.target_effector_pose)

        def _set_state_safe(obj_state, obj_id):
            if obj_state is not None:
                assert obj_id is not None, "Cannot set state for missing object."
                obj_state.set_bullet_state(self._pybullet_client, obj_id)
            else:
                assert obj_id is None, f"No state found for obj_id {obj_id}"

        robot_end_effectors = state["robot_end_effectors"]
        _set_state_safe(
            None if not robot_end_effectors else robot_end_effectors[0],
            self.robot.end_effector,
        )

        for target_state, target_id in zip(state["targets"], self._target_ids):
            _set_state_safe(target_state, target_id)

        obj_ids = self.get_obj_ids()
        assert len(state["objects"]) == len(obj_ids), "State length mismatch"
        for obj_state, obj_id in zip(state["objects"], obj_ids):
            _set_state_safe(obj_state, obj_id)

        self.reset(reset_poses=False)


class BlockPushHorizontalMultimodal(BlockPushMultimodal):
    def _reset_object_poses(self, workspace_center_x, workspace_center_y):
        # Reset block poses.
        self._reset_block_poses(workspace_center_y)

        # Reset target poses.
        self._reset_target_poses(workspace_center_y)

    def _reset_block_poses(self, workspace_center_y):
        """Resets block poses."""

        # Helper for choosing random block position.
        def _reset_block_pose(idx, add=0.0, avoid=None):
            def _get_random_translation():
                block_x = 0.35 + 0.5 * self._rng.uniform(
                    low=-RANDOM_X_SHIFT, high=RANDOM_X_SHIFT
                )
                block_y = (
                    workspace_center_y
                    + add
                    + 0.5 * self._rng.uniform(low=-RANDOM_Y_SHIFT, high=RANDOM_Y_SHIFT)
                )
                block_translation = np.array([block_x, block_y, 0])
                return block_translation

            if avoid is None:
                block_translation = _get_random_translation()
            else:
                # Reject targets too close to `avoid`.
                for _ in range(NUM_RESET_ATTEMPTS):
                    block_translation = _get_random_translation()
                    dist = np.linalg.norm(block_translation[0] - avoid[0])
                    # print('block inner try_idx %d, dist %.3f' % (try_idx, dist))
                    if dist > MIN_BLOCK_DIST:
                        break
            block_sampled_angle = self._rng.uniform(math.pi)
            block_rotation = transform.Rotation.from_rotvec([0, 0, block_sampled_angle])
            self._pybullet_client.resetBasePositionAndOrientation(
                self._block_ids[idx],
                block_translation.tolist(),
                block_rotation.as_quat().tolist(),
            )
            return block_translation

        # Reject targets too close to `avoid`.
        for _ in range(NUM_RESET_ATTEMPTS):
            # Reset first block.
            add = 0.2 * self._rng.choice([-1, 1])
            b0_translation = _reset_block_pose(0, add=add)
            # Reset second block away from first block.
            b1_translation = _reset_block_pose(1, add=-add, avoid=b0_translation)
            dist = np.linalg.norm(b0_translation[0] - b1_translation[0])
            if dist > MIN_BLOCK_DIST:
                break
        else:
            raise ValueError("could not find matching block")
        assert dist > MIN_BLOCK_DIST

    def _reset_target_poses(self, workspace_center_y):
        """Resets target poses."""

        def _reset_target_pose(idx, add=0.0, avoid=None):
            def _get_random_translation():
                # Choose x,y randomly.
                target_x = 0.5 + self._rng.uniform(
                    low=-0.05 * RANDOM_X_SHIFT, high=0.05 * RANDOM_X_SHIFT
                )
                target_y = (
                    workspace_center_y
                    + add
                    + self._rng.uniform(
                        low=-0.05 * RANDOM_Y_SHIFT, high=0.05 * RANDOM_Y_SHIFT
                    )
                )
                target_translation = np.array([target_x, target_y, 0.020])
                return target_translation

            if avoid is None:
                target_translation = _get_random_translation()
            else:
                # Reject targets too close to `avoid`.
                for _ in range(NUM_RESET_ATTEMPTS):
                    target_translation = _get_random_translation()
                    dist = np.linalg.norm(target_translation[0] - avoid[0])
                    # print('target inner try_idx %d, dist %.3f' % (try_idx, dist))
                    if dist > MIN_TARGET_DIST:
                        break
            target_sampled_angle = math.pi + self._rng.uniform(
                low=-math.pi / 30, high=math.pi / 30
            )
            target_rotation = transform.Rotation.from_rotvec(
                [0, 0, target_sampled_angle]
            )
            self._pybullet_client.resetBasePositionAndOrientation(
                self._target_ids[idx],
                target_translation.tolist(),
                target_rotation.as_quat().tolist(),
            )
            self._target_poses[idx] = Pose3d(
                rotation=target_rotation, translation=target_translation
            )

        if self._target_poses is None:
            self._target_poses = [None for _ in range(len(self._target_ids))]

        for _ in range(NUM_RESET_ATTEMPTS):
            # Choose the first target.
            add = 0.2 * self._rng.choice([-1, 1])
            # Randomly flip the location of the targets.
            _reset_target_pose(0, add=add)
            _reset_target_pose(1, add=-add, avoid=self._target_poses[0].translation)
            dist = np.linalg.norm(
                self._target_poses[0].translation[0]
                - self._target_poses[1].translation[0]
            )
            break
            # if dist > MIN_TARGET_DIST:
            #     break
        else:
            raise ValueError("could not find matching target")
        # assert dist > MIN_TARGET_DIST


if "BlockPushMultimodal-v0" in registration.registry.env_specs:
    del registration.registry.env_specs["BlockPushMultimodal-v0"]

registration.register(
    id="BlockPushMultimodal-v0", entry_point=BlockPushMultimodal, max_episode_steps=350
)

registration.register(
    id="BlockPushMultimodalFlipped-v0",
    entry_point=BlockPushHorizontalMultimodal,
    max_episode_steps=25,
)

registration.register(
    id="SharedBlockPushMultimodal-v0",
    entry_point=BlockPushMultimodal,
    kwargs=dict(shared_memory=True),
    max_episode_steps=350,
)
registration.register(
    id="BlockPushMultimodalRgb-v0",
    entry_point=BlockPushMultimodal,
    max_episode_steps=350,
    kwargs=dict(image_size=(block_pushing.IMAGE_HEIGHT, block_pushing.IMAGE_WIDTH)),
)
