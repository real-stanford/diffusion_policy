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

"""Discontinuous block pushing."""
import collections
import enum
import math
from typing import List, Optional

from gym import spaces
from gym.envs import registration
from diffusion_policy.env.block_pushing import block_pushing
from diffusion_policy.env.block_pushing.utils import utils_pybullet
from diffusion_policy.env.block_pushing.utils.pose3d import Pose3d
import numpy as np
from scipy.spatial import transform
import pybullet
import pybullet_utils.bullet_client as bullet_client

ZONE2_URDF_PATH = "third_party/py/envs/assets/zone2.urdf"

MIN_TARGET_DIST = 0.15
NUM_RESET_ATTEMPTS = 1000


def build_env_name(task, shared_memory, use_image_obs):
    """Construct the env name from parameters."""
    del task
    env_name = "BlockPushDiscontinuous"

    if use_image_obs:
        env_name = env_name + "Rgb"

    if shared_memory:
        env_name = "Shared" + env_name

    env_name = env_name + "-v0"

    return env_name


class BlockTaskVariant(enum.Enum):
    REACH = "Reach"
    REACH_NORMALIZED = "ReachNormalized"
    PUSH = "Push"
    PUSH_NORMALIZED = "PushNormalized"
    INSERT = "Insert"


# pytype: skip-file
class BlockPushDiscontinuous(block_pushing.BlockPush):
    """Discontinuous block pushing."""

    def __init__(
        self,
        control_frequency=10.0,
        task=BlockTaskVariant.PUSH,
        image_size=None,
        shared_memory=False,
        seed=None,
        goal_dist_tolerance=0.04,
    ):
        super(BlockPushDiscontinuous, self).__init__(
            control_frequency=control_frequency,
            task=task,
            image_size=image_size,
            shared_memory=shared_memory,
            seed=seed,
            goal_dist_tolerance=goal_dist_tolerance,
        )

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
        target_urdf_path = block_pushing.ZONE_URDF_PATH

        self._target_ids = []
        for _ in [block_pushing.ZONE_URDF_PATH, ZONE2_URDF_PATH]:
            self._target_ids.append(
                utils_pybullet.load_urdf(
                    self._pybullet_client, target_urdf_path, useFixedBase=True
                )
            )
        self._block_ids = [
            utils_pybullet.load_urdf(
                self._pybullet_client, block_pushing.BLOCK_URDF_PATH, useFixedBase=False
            )
        ]

        # Re-enable rendering.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

        self.step_simulation_to_stabilize()

    def _reset_target_poses(self, workspace_center_x):
        """Resets target poses."""
        self._target_poses = [None for _ in range(len(self._target_ids))]

        def _reset_target_pose(idx, avoid=None):
            def _get_random_translation():
                # Choose x,y randomly.
                target_x = workspace_center_x + self._rng.uniform(low=-0.10, high=0.10)
                # Fix ys for this environment.
                if idx == 0:
                    target_y = 0
                else:
                    target_y = 0.4
                target_translation = np.array([target_x, target_y, 0.020])
                return target_translation

            if avoid is None:
                target_translation = _get_random_translation()
            else:
                # Reject targets too close to `avoid`.
                for _ in range(NUM_RESET_ATTEMPTS):
                    target_translation = _get_random_translation()
                    dist = np.linalg.norm(target_translation[0] - avoid[0])
                    if dist > MIN_TARGET_DIST:
                        break
            target_sampled_angle = math.pi + self._rng.uniform(
                low=-math.pi / 6, high=math.pi / 6
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

        try_idx = 0
        while True:
            # Choose the first target.
            _reset_target_pose(0)
            # Choose the second target, avoiding the first.
            _reset_target_pose(1, avoid=self._target_poses[0].translation)
            dist = np.linalg.norm(
                self._target_poses[0].translation[0]
                - self._target_poses[1].translation[0]
            )
            if dist > MIN_TARGET_DIST:
                break
            try_idx += 1
            if try_idx >= NUM_RESET_ATTEMPTS:
                raise ValueError("could not find matching target")
        assert dist > MIN_TARGET_DIST

    def reset(self):
        self._pybullet_client.restoreState(self._saved_state)

        rotation = transform.Rotation.from_rotvec([0, math.pi, 0])
        translation = np.array([0.3, -0.4, block_pushing.EFFECTOR_HEIGHT])
        starting_pose = Pose3d(rotation=rotation, translation=translation)
        self._set_robot_target_effector_pose(starting_pose)

        workspace_center_x = 0.4

        # Reset block pose.
        block_x = workspace_center_x + self._rng.uniform(low=-0.1, high=0.1)
        block_y = -0.2 + self._rng.uniform(low=-0.15, high=0.15)
        block_translation = np.array([block_x, block_y, 0])
        block_sampled_angle = self._rng.uniform(math.pi)
        block_rotation = transform.Rotation.from_rotvec([0, 0, block_sampled_angle])

        self._pybullet_client.resetBasePositionAndOrientation(
            self._block_ids[0],
            block_translation.tolist(),
            block_rotation.as_quat().tolist(),
        )

        # Reset target pose.
        self._reset_target_poses(workspace_center_x)

        self.step_simulation_to_stabilize()
        state = self._compute_state()
        self._previous_state = state
        self.min_dist_to_first_goal = np.inf
        self.min_dist_to_second_goal = np.inf
        self.steps = 0
        return state

    def _compute_goal_distance(self, state):
        # Reward is 1. blocks is inside any target.
        return np.mean([self.min_dist_to_first_goal, self.min_dist_to_second_goal])

    def _compute_state(self):
        effector_pose = self._robot.forward_kinematics()
        block_position_and_orientation = (
            self._pybullet_client.getBasePositionAndOrientation(self._block_ids[0])
        )
        block_pose = Pose3d(
            rotation=transform.Rotation.from_quat(block_position_and_orientation[1]),
            translation=block_position_and_orientation[0],
        )

        def _yaw_from_pose(pose):
            return np.array([pose.rotation.as_euler("xyz", degrees=False)[-1]])

        obs = collections.OrderedDict(
            block_translation=block_pose.translation[0:2],
            block_orientation=_yaw_from_pose(block_pose),
            effector_translation=effector_pose.translation[0:2],
            effector_target_translation=self._target_effector_pose.translation[0:2],
            target_translation=self._target_poses[0].translation[0:2],
            target_orientation=_yaw_from_pose(self._target_poses[0]),
            target2_translation=self._target_poses[1].translation[0:2],
            target2_orientation=_yaw_from_pose(self._target_poses[1]),
        )
        if self._image_size is not None:
            obs["rgb"] = self._render_camera(self._image_size)
        return obs

    def step(self, action):
        self._step_robot_and_sim(action)
        state = self._compute_state()
        reward = self._get_reward(state)
        done = False
        if reward > 0.0:
            done = True
        # Cache so we can compute success.
        self.state = state
        return state, reward, done, {}

    def dist(self, state, target):
        # Reward is 1. blocks is inside any target.
        return np.linalg.norm(
            state["block_translation"] - state["%s_translation" % target]
        )

    def _get_reward(self, state):
        """Reward is 1.0 if agent hits both goals and stays at second."""
        # This also statefully updates these values.
        self.min_dist_to_first_goal = min(
            self.dist(state, "target"), self.min_dist_to_first_goal
        )
        self.min_dist_to_second_goal = min(
            self.dist(state, "target2"), self.min_dist_to_second_goal
        )

        def _reward(thresh):
            reward_first = True if self.min_dist_to_first_goal < thresh else False
            reward_second = True if self.min_dist_to_second_goal < thresh else False
            return 1.0 if (reward_first and reward_second) else 0.0

        reward = _reward(self.goal_dist_tolerance)
        return reward

    @property
    def succeeded(self):
        thresh = self.goal_dist_tolerance
        hit_first = True if self.min_dist_to_first_goal < thresh else False
        hit_second = True if self.min_dist_to_first_goal < thresh else False
        current_distance_to_second = self.dist(self.state, "target2")
        still_at_second = True if current_distance_to_second < thresh else False
        return hit_first and hit_second and still_at_second

    def _create_observation_space(self, image_size):
        pi2 = math.pi * 2

        obs_dict = collections.OrderedDict(
            block_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            block_orientation=spaces.Box(low=-pi2, high=pi2, shape=(1,)),  # phi
            effector_translation=spaces.Box(
                # Small buffer for to IK noise.
                low=block_pushing.WORKSPACE_BOUNDS[0] - 0.1,
                high=block_pushing.WORKSPACE_BOUNDS[1] + 0.1,
            ),  # x,y
            effector_target_translation=spaces.Box(
                # Small buffer for to IK noise.
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


if "BlockPushDiscontinuous-v0" in registration.registry.env_specs:
    del registration.registry.env_specs["BlockPushDiscontinuous-v0"]

registration.register(
    id="BlockPushDiscontinuous-v0",
    entry_point=BlockPushDiscontinuous,
    max_episode_steps=200,
)

registration.register(
    id="BlockPushDiscontinuousRgb-v0",
    entry_point=BlockPushDiscontinuous,
    max_episode_steps=200,
    kwargs=dict(image_size=(block_pushing.IMAGE_HEIGHT, block_pushing.IMAGE_WIDTH)),
)
