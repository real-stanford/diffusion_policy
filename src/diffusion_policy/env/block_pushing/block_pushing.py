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

"""Simple block environments for the XArm."""

import collections
import enum
import math
import time
from typing import Dict, List, Optional, Tuple, Union

import gym
from gym import spaces
from gym.envs import registration
from diffusion_policy.env.block_pushing.utils import utils_pybullet
from diffusion_policy.env.block_pushing.utils import xarm_sim_robot
from diffusion_policy.env.block_pushing.utils.pose3d import Pose3d
from diffusion_policy.env.block_pushing.utils.utils_pybullet import ObjState
from diffusion_policy.env.block_pushing.utils.utils_pybullet import XarmState
import numpy as np
from scipy.spatial import transform
import pybullet
import pybullet_utils.bullet_client as bullet_client

import matplotlib.pyplot as plt

BLOCK_URDF_PATH = "third_party/py/envs/assets/block.urdf"
PLANE_URDF_PATH = "third_party/bullet/examples/pybullet/gym/pybullet_data/" "plane.urdf"
WORKSPACE_URDF_PATH = "third_party/py/envs/assets/workspace.urdf"
ZONE_URDF_PATH = "third_party/py/envs/assets/zone.urdf"
INSERT_URDF_PATH = "third_party/py/envs/assets/insert.urdf"

EFFECTOR_HEIGHT = 0.06
EFFECTOR_DOWN_ROTATION = transform.Rotation.from_rotvec([0, math.pi, 0])

WORKSPACE_BOUNDS = np.array(((0.15, -0.5), (0.7, 0.5)))

# Min/max bounds calculated from oracle data using:
# ibc/environments/board2d_dataset_statistics.ipynb
# to calculate [mean - 3 * std, mean + 3 * std] using the oracle data.
# pylint: disable=line-too-long
ACTION_MIN = np.array([-0.02547718, -0.02090043], np.float32)
ACTION_MAX = np.array([0.02869084, 0.04272365], np.float32)
EFFECTOR_TARGET_TRANSLATION_MIN = np.array(
    [0.1774151772260666, -0.6287994794547558], np.float32
)
EFFECTOR_TARGET_TRANSLATION_MAX = np.array(
    [0.5654461532831192, 0.5441607423126698], np.float32
)
EFFECTOR_TARGET_TO_BLOCK_TRANSLATION_MIN = np.array(
    [-0.07369826920330524, -0.11395704373717308], np.float32
)
EFFECTOR_TARGET_TO_BLOCK_TRANSLATION_MAX = np.array(
    [0.10131562314927578, 0.19391131028532982], np.float32
)
EFFECTOR_TARGET_TO_TARGET_TRANSLATION_MIN = np.array(
    [-0.17813862301409245, -0.3309651017189026], np.float32
)
EFFECTOR_TARGET_TO_TARGET_TRANSLATION_MAX = np.array(
    [0.23726161383092403, 0.8404090404510498], np.float32
)
BLOCK_ORIENTATION_COS_SIN_MIN = np.array(
    [-2.0649861991405487, -0.6154364347457886], np.float32
)
BLOCK_ORIENTATION_COS_SIN_MAX = np.array(
    [1.6590178310871124, 1.8811014890670776], np.float32
)
TARGET_ORIENTATION_COS_SIN_MIN = np.array(
    [-1.0761439241468906, -0.8846937336493284], np.float32
)
TARGET_ORIENTATION_COS_SIN_MAX = np.array(
    [-0.8344330154359341, 0.8786859593819827], np.float32
)

# Hardcoded Pose joints to make sure we don't have surprises from using the
# IK solver on reset. The joint poses correspond to the Pose with:
#   rotation = rotation3.Rotation3.from_axis_angle([0, 1, 0], math.pi)
#   translation = np.array([0.3, -0.4, 0.07])
INITIAL_JOINT_POSITIONS = np.array(
    [
        -0.9254632489674508,
        0.6990770671568564,
        -1.106629064060494,
        0.0006653351931553931,
        0.3987969742311386,
        -4.063402065624296,
    ]
)

DEFAULT_CAMERA_POSE = (1.0, 0, 0.75)
DEFAULT_CAMERA_ORIENTATION = (np.pi / 4, np.pi, -np.pi / 2)
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
CAMERA_INTRINSICS = (
    0.803 * IMAGE_WIDTH,  # fx
    0,
    IMAGE_WIDTH / 2.0,  # cx
    0,
    0.803 * IMAGE_WIDTH,  # fy
    IMAGE_HEIGHT / 2.0,  # cy
    0,
    0,
    1,
)

# "Realistic" visuals.
X_MIN_REAL = 0.15
X_MAX_REAL = 0.6
Y_MIN_REAL = -0.3048
Y_MAX_REAL = 0.3048
WORKSPACE_BOUNDS_REAL = np.array(((X_MIN_REAL, Y_MIN_REAL), (X_MAX_REAL, Y_MAX_REAL)))
WORKSPACE_URDF_PATH_REAL = "third_party/py/ibc/environments/assets/workspace_real.urdf"
CAMERA_POSE_REAL = (0.75, 0, 0.5)
CAMERA_ORIENTATION_REAL = (np.pi / 5, np.pi, -np.pi / 2)

IMAGE_WIDTH_REAL = 320
IMAGE_HEIGHT_REAL = 180
CAMERA_INTRINSICS_REAL = (
    0.803 * IMAGE_WIDTH_REAL,  # fx
    0,
    IMAGE_WIDTH_REAL / 2.0,  # cx
    0,
    0.803 * IMAGE_WIDTH_REAL,  # fy
    IMAGE_HEIGHT_REAL / 2.0,  # cy
    0,
    0,
    1,
)
# pylint: enable=line-too-long


def build_env_name(task, shared_memory, use_image_obs, use_normalized_env=False):
    """Construct the env name from parameters."""
    if isinstance(task, str):
        task = BlockTaskVariant[task]
    env_name = "Block" + task.value

    if use_image_obs:
        env_name = env_name + "Rgb"
    if use_normalized_env:
        env_name = env_name + "Normalized"
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


def sleep_spin(sleep_time_sec):
    """Spin wait sleep. Avoids time.sleep accuracy issues on Windows."""
    if sleep_time_sec <= 0:
        return
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sleep_time_sec:
        pass


class BlockPush(gym.Env):
    """Simple XArm environment for block pushing."""

    def __init__(
        self,
        control_frequency=10.0,
        task=BlockTaskVariant.PUSH,
        image_size=None,
        shared_memory=False,
        seed=None,
        goal_dist_tolerance=0.01,
        effector_height=None,
        visuals_mode="default",
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
          effector_height: float, custom height for end effector.
          visuals_mode: 'default' or 'real'.
        """
        # pybullet.connect(pybullet.GUI)
        # pybullet.resetDebugVisualizerCamera(
        #     cameraDistance=1.5,
        #     cameraYaw=0,
        #     cameraPitch=-40,
        #     cameraTargetPosition=[0.55, -0.35, 0.2],
        # )
        if visuals_mode != "default" and visuals_mode != "real":
            raise ValueError("visuals_mode must be `real` or `default`.")
        self._task = task
        self._connection_mode = pybullet.DIRECT
        if shared_memory:
            self._connection_mode = pybullet.SHARED_MEMORY

        self.goal_dist_tolerance = goal_dist_tolerance

        self.effector_height = effector_height or EFFECTOR_HEIGHT

        self._visuals_mode = visuals_mode
        if visuals_mode == "default":
            self._camera_pose = DEFAULT_CAMERA_POSE
            self._camera_orientation = DEFAULT_CAMERA_ORIENTATION
            self.workspace_bounds = WORKSPACE_BOUNDS
            self._image_size = image_size
            self._camera_instrinsics = CAMERA_INTRINSICS
            self._workspace_urdf_path = WORKSPACE_URDF_PATH
        else:
            self._camera_pose = CAMERA_POSE_REAL
            self._camera_orientation = CAMERA_ORIENTATION_REAL
            self.workspace_bounds = WORKSPACE_BOUNDS_REAL
            self._image_size = image_size
            self._camera_instrinsics = CAMERA_INTRINSICS_REAL
            self._workspace_urdf_path = WORKSPACE_URDF_PATH_REAL

        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))  # x, y
        self.observation_space = self._create_observation_space(image_size)

        self._rng = np.random.RandomState(seed=seed)
        self._block_ids = None
        self._previous_state = None
        self._robot = None
        self._workspace_uid = None
        self._target_id = None
        self._target_pose = None
        self._target_effector_pose = None
        self._pybullet_client = None
        self.reach_target_translation = None
        self._setup_pybullet_scene()
        self._saved_state = None

        assert isinstance(self._pybullet_client, bullet_client.BulletClient)
        self._control_frequency = control_frequency
        self._step_frequency = (
            1 / self._pybullet_client.getPhysicsEngineParameters()["fixedTimeStep"]
        )

        self._last_loop_time = None
        self._last_loop_frame_sleep_time = None
        if self._step_frequency % self._control_frequency != 0:
            raise ValueError(
                "Control frequency should be a multiple of the "
                "configured Bullet TimeStep."
            )
        self._sim_steps_per_step = int(self._step_frequency / self._control_frequency)

        self.rendered_img = None
        self._abs_action = abs_action

        # Use saved_state and restore to make reset safe as no simulation state has
        # been updated at this state, but the assets are now loaded.
        self.save_state()
        self.reset()

    @property
    def pybullet_client(self):
        return self._pybullet_client

    @property
    def robot(self):
        return self._robot

    @property
    def workspace_uid(self):
        return self._workspace_uid

    @property
    def target_effector_pose(self):
        return self._target_effector_pose

    @property
    def target_pose(self):
        return self._target_pose

    @property
    def control_frequency(self):
        return self._control_frequency

    @property
    def connection_mode(self):
        return self._connection_mode

    def save_state(self):
        self._saved_state = self._pybullet_client.saveState()

    def set_goal_dist_tolerance(self, val):
        self.goal_dist_tolerance = val

    def get_control_frequency(self):
        return self._control_frequency

    def compute_state(self):
        return self._compute_state()

    def get_goal_translation(self):
        """Return the translation component of the goal (2D)."""
        if self._task == BlockTaskVariant.REACH:
            return np.concatenate([self.reach_target_translation, [0]])
        else:
            return self._target_pose.translation if self._target_pose else None

    def get_obj_ids(self):
        return self._block_ids

    def _setup_workspace_and_robot(self, end_effector="suction"):
        self._pybullet_client.resetSimulation()
        self._pybullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        self._pybullet_client.setPhysicsEngineParameter(enableFileCaching=0)
        self._pybullet_client.setGravity(0, 0, -9.8)

        utils_pybullet.load_urdf(
            self._pybullet_client, PLANE_URDF_PATH, basePosition=[0, 0, -0.001]
        )
        self._workspace_uid = utils_pybullet.load_urdf(
            self._pybullet_client,
            self._workspace_urdf_path,
            basePosition=[0.35, 0, 0.0],
        )

        self._robot = xarm_sim_robot.XArmSimRobot(
            self._pybullet_client,
            initial_joint_positions=INITIAL_JOINT_POSITIONS,
            end_effector=end_effector,
            color="white" if self._visuals_mode == "real" else "default",
        )

    def _setup_pybullet_scene(self):
        self._pybullet_client = bullet_client.BulletClient(self._connection_mode)

        # Temporarily disable rendering to speed up loading URDFs.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        self._setup_workspace_and_robot()

        if self._task == BlockTaskVariant.INSERT:
            target_urdf_path = INSERT_URDF_PATH
        else:
            target_urdf_path = ZONE_URDF_PATH

        self._target_id = utils_pybullet.load_urdf(
            self._pybullet_client, target_urdf_path, useFixedBase=True
        )
        self._block_ids = [
            utils_pybullet.load_urdf(
                self._pybullet_client, BLOCK_URDF_PATH, useFixedBase=False
            )
        ]

        # Re-enable rendering.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

        self.step_simulation_to_stabilize()

    def step_simulation_to_stabilize(self, nsteps=100):
        for _ in range(nsteps):
            self._pybullet_client.stepSimulation()

    def seed(self, seed=None):
        self._rng = np.random.RandomState(seed=seed)

    def _set_robot_target_effector_pose(self, pose):
        self._target_effector_pose = pose
        self._robot.set_target_effector_pose(pose)

    def reset(self, reset_poses=True):
        workspace_center_x = 0.4

        if reset_poses:
            self._pybullet_client.restoreState(self._saved_state)

            rotation = transform.Rotation.from_rotvec([0, math.pi, 0])
            translation = np.array([0.3, -0.4, self.effector_height])
            starting_pose = Pose3d(rotation=rotation, translation=translation)
            self._set_robot_target_effector_pose(starting_pose)

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
            target_x = workspace_center_x + self._rng.uniform(low=-0.10, high=0.10)
            target_y = 0.2 + self._rng.uniform(low=-0.15, high=0.15)
            target_translation = np.array([target_x, target_y, 0.020])

            target_sampled_angle = math.pi + self._rng.uniform(
                low=-math.pi / 6, high=math.pi / 6
            )
            target_rotation = transform.Rotation.from_rotvec(
                [0, 0, target_sampled_angle]
            )

            self._pybullet_client.resetBasePositionAndOrientation(
                self._target_id,
                target_translation.tolist(),
                target_rotation.as_quat().tolist(),
            )
        else:
            (
                target_translation,
                target_orientation_quat,
            ) = self._pybullet_client.getBasePositionAndOrientation(self._target_id)
            target_rotation = transform.Rotation.from_quat(target_orientation_quat)
            target_translation = np.array(target_translation)

        self._target_pose = Pose3d(
            rotation=target_rotation, translation=target_translation
        )

        if reset_poses:
            self.step_simulation_to_stabilize()

        state = self._compute_state()
        self._previous_state = state

        if self._task == BlockTaskVariant.REACH:
            self._compute_reach_target(state)

        self._init_goal_distance = self._compute_goal_distance(state)
        init_goal_eps = 1e-7
        assert self._init_goal_distance > init_goal_eps
        self.best_fraction_reduced_goal_dist = 0.0

        return state

    def _compute_goal_distance(self, state):
        goal_translation = self.get_goal_translation()
        if self._task != BlockTaskVariant.REACH:
            goal_distance = np.linalg.norm(
                state["block_translation"] - goal_translation[0:2]
            )
        else:
            goal_distance = np.linalg.norm(
                state["effector_translation"] - goal_translation[0:2]
            )
        return goal_distance

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
            target_translation=self._target_pose.translation[0:2],
            target_orientation=_yaw_from_pose(self._target_pose),
        )
        if self._image_size is not None:
            obs["rgb"] = self._render_camera(self._image_size)
        return obs

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
            rotation=EFFECTOR_DOWN_ROTATION, translation=target_effector_translation
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
                sleep_spin(frame_sleep_time)
            self._pybullet_client.stepSimulation()

    def step(self, action):
        self._step_robot_and_sim(action)

        state = self._compute_state()

        goal_distance = self._compute_goal_distance(state)
        fraction_reduced_goal_distance = 1.0 - (
            goal_distance / self._init_goal_distance
        )
        if fraction_reduced_goal_distance > self.best_fraction_reduced_goal_dist:
            self.best_fraction_reduced_goal_dist = fraction_reduced_goal_distance

        done = False
        reward = self.best_fraction_reduced_goal_dist

        # Terminate the episode if the block is close enough to the target.
        if goal_distance < self.goal_dist_tolerance:
            reward = 1.0
            done = True

        return state, reward, done, {}

    @property
    def succeeded(self):
        state = self._compute_state()
        goal_distance = self._compute_goal_distance(state)
        if goal_distance < self.goal_dist_tolerance:
            return True
        return False

    @property
    def goal_distance(self):
        state = self._compute_state()
        return self._compute_goal_distance(state)

    def render(self, mode="rgb_array"):
        if self._image_size is not None:
            image_size = self._image_size
        else:
            # This allows rendering even for state-only obs,
            # for visualization.
            image_size = (IMAGE_HEIGHT, IMAGE_WIDTH)

        data = self._render_camera(image_size=(image_size[0], image_size[1]))
        if mode == "human":
            if self.rendered_img is None:
                self.rendered_img = plt.imshow(
                    np.zeros((image_size[0], image_size[1], 4))
                )
            else:
                self.rendered_img.set_data(data)
            plt.draw()
            plt.pause(0.00001)
        return data

    def close(self):
        self._pybullet_client.disconnect()

    def calc_camera_params(self, image_size):
        # Mimic RealSense D415 camera parameters.
        intrinsics = self._camera_instrinsics

        # Set default camera poses.
        front_position = self._camera_pose
        front_rotation = self._camera_orientation
        front_rotation = self._pybullet_client.getQuaternionFromEuler(front_rotation)
        # Default camera configs.
        zrange = (0.01, 10.0)

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = self._pybullet_client.getMatrixFromQuaternion(front_rotation)
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = front_position + lookdir
        focal_len = intrinsics[0]
        znear, zfar = zrange
        viewm = self._pybullet_client.computeViewMatrix(front_position, lookat, updir)
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = image_size[1] / image_size[0]
        projm = self._pybullet_client.computeProjectionMatrixFOV(
            fovh, aspect_ratio, znear, zfar
        )

        return viewm, projm, front_position, lookat, updir

    def _render_camera(self, image_size):
        """Render RGB image with RealSense configuration."""
        viewm, projm, _, _, _ = self.calc_camera_params(image_size)

        # Render with OpenGL camera settings.
        _, _, color, _, _ = self._pybullet_client.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        )

        # Get color image.
        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel

        return color.astype(np.uint8)

    def _create_observation_space(self, image_size):
        pi2 = math.pi * 2

        obs_dict = collections.OrderedDict(
            block_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            block_orientation=spaces.Box(low=-pi2, high=pi2, shape=(1,)),  # phi
            effector_translation=spaces.Box(
                low=self.workspace_bounds[0] - 0.1,  # Small buffer for to IK noise.
                high=self.workspace_bounds[1] + 0.1,
            ),  # x,y
            effector_target_translation=spaces.Box(
                low=self.workspace_bounds[0] - 0.1,  # Small buffer for to IK noise.
                high=self.workspace_bounds[1] + 0.1,
            ),  # x,y
            target_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            target_orientation=spaces.Box(
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
                goal_translation=self.get_goal_translation(),
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
        if self._target_id:
            state["targets"].append(
                ObjState.get_bullet_state(self._pybullet_client, self._target_id)
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

        targets = state["targets"]
        _set_state_safe(None if not targets else targets[0], self._target_id)

        obj_ids = self.get_obj_ids()
        assert len(state["objects"]) == len(obj_ids), "State length mismatch"
        for obj_state, obj_id in zip(state["objects"], obj_ids):
            _set_state_safe(obj_state, obj_id)

        self.reset(reset_poses=False)


class BlockPushNormalized(gym.Env):
    """Simple XArm environment for block pushing, normalized state and actions."""

    def __init__(
        self,
        control_frequency=10.0,
        task=BlockTaskVariant.PUSH_NORMALIZED,
        image_size=None,
        shared_memory=False,
        seed=None,
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
        """
        # Map normalized task to unnormalized task.
        if task == BlockTaskVariant.PUSH_NORMALIZED:
            env_task = BlockTaskVariant.PUSH
        elif task == BlockTaskVariant.REACH_NORMALIZED:
            env_task = BlockTaskVariant.REACH
        else:
            raise ValueError("Unsupported task %s" % str(task))
        self._env = BlockPush(
            control_frequency, env_task, image_size, shared_memory, seed
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Dict(
            collections.OrderedDict(
                effector_target_translation=spaces.Box(low=-1, high=1, shape=(2,)),
                effector_target_to_block_translation=spaces.Box(
                    low=-1, high=1, shape=(2,)
                ),
                block_orientation_cos_sin=spaces.Box(low=-1, high=1, shape=(2,)),
                effector_target_to_target_translation=spaces.Box(
                    low=-1, high=1, shape=(2,)
                ),
                target_orientation_cos_sin=spaces.Box(low=-1, high=1, shape=(2,)),
            )
        )
        self.reset()

    def get_control_frequency(self):
        return self._env.get_control_frequency()

    @property
    def reach_target_translation(self):
        return self._env.reach_target_translation

    def seed(self, seed=None):
        self._env.seed(seed)

    def reset(self):
        state = self._env.reset()
        return self.calc_normalized_state(state)

    def step(self, action):
        # The environment is normalized [mean-3*std, mean+3*std] -> [-1, 1].
        action = np.clip(action, a_min=-1.0, a_max=1.0)
        state, reward, done, info = self._env.step(
            self.calc_unnormalized_action(action)
        )
        state = self.calc_normalized_state(state)
        reward = reward * 100  # Keep returns in [0, 100]
        return state, reward, done, info

    def render(self, mode="rgb_array"):
        return self._env.render(mode)

    def close(self):
        self._env.close()

    @staticmethod
    def _normalize(values, values_min, values_max):
        offset = (values_max + values_min) * 0.5
        scale = (values_max - values_min) * 0.5
        return (values - offset) / scale  # [min, max] -> [-1, 1]

    @staticmethod
    def _unnormalize(values, values_min, values_max):
        offset = (values_max + values_min) * 0.5
        scale = (values_max - values_min) * 0.5
        return values * scale + offset  # [-1, 1] -> [min, max]

    @classmethod
    def calc_normalized_action(cls, action):
        return cls._normalize(action, ACTION_MIN, ACTION_MAX)

    @classmethod
    def calc_unnormalized_action(cls, norm_action):
        return cls._unnormalize(norm_action, ACTION_MIN, ACTION_MAX)

    @classmethod
    def calc_normalized_state(cls, state):

        effector_target_translation = cls._normalize(
            state["effector_target_translation"],
            EFFECTOR_TARGET_TRANSLATION_MIN,
            EFFECTOR_TARGET_TRANSLATION_MAX,
        )

        effector_target_to_block_translation = cls._normalize(
            state["block_translation"] - state["effector_target_translation"],
            EFFECTOR_TARGET_TO_BLOCK_TRANSLATION_MIN,
            EFFECTOR_TARGET_TO_BLOCK_TRANSLATION_MAX,
        )
        ori = state["block_orientation"][0]
        block_orientation_cos_sin = cls._normalize(
            np.array([math.cos(ori), math.sin(ori)], np.float32),
            BLOCK_ORIENTATION_COS_SIN_MIN,
            BLOCK_ORIENTATION_COS_SIN_MAX,
        )

        effector_target_to_target_translation = cls._normalize(
            state["target_translation"] - state["effector_target_translation"],
            EFFECTOR_TARGET_TO_TARGET_TRANSLATION_MIN,
            EFFECTOR_TARGET_TO_TARGET_TRANSLATION_MAX,
        )
        ori = state["target_orientation"][0]
        target_orientation_cos_sin = cls._normalize(
            np.array([math.cos(ori), math.sin(ori)], np.float32),
            TARGET_ORIENTATION_COS_SIN_MIN,
            TARGET_ORIENTATION_COS_SIN_MAX,
        )

        # Note: We do not include effector_translation in the normalized state.
        # This means the unnormalized -> normalized mapping is not invertable.
        return collections.OrderedDict(
            effector_target_translation=effector_target_translation,
            effector_target_to_block_translation=effector_target_to_block_translation,
            block_orientation_cos_sin=block_orientation_cos_sin,
            effector_target_to_target_translation=effector_target_to_target_translation,
            target_orientation_cos_sin=target_orientation_cos_sin,
        )

    @classmethod
    def calc_unnormalized_state(cls, norm_state):

        effector_target_translation = cls._unnormalize(
            norm_state["effector_target_translation"],
            EFFECTOR_TARGET_TRANSLATION_MIN,
            EFFECTOR_TARGET_TRANSLATION_MAX,
        )
        # Note: normalized state does not include effector_translation state, this
        # means this component will be missing (and is marked nan).
        effector_translation = np.array([np.nan, np.nan], np.float32)

        effector_target_to_block_translation = cls._unnormalize(
            norm_state["effector_target_to_block_translation"],
            EFFECTOR_TARGET_TO_BLOCK_TRANSLATION_MIN,
            EFFECTOR_TARGET_TO_BLOCK_TRANSLATION_MAX,
        )
        block_translation = (
            effector_target_to_block_translation + effector_target_translation
        )
        ori_cos_sin = cls._unnormalize(
            norm_state["block_orientation_cos_sin"],
            BLOCK_ORIENTATION_COS_SIN_MIN,
            BLOCK_ORIENTATION_COS_SIN_MAX,
        )
        block_orientation = np.array(
            [math.atan2(ori_cos_sin[1], ori_cos_sin[0])], np.float32
        )

        effector_target_to_target_translation = cls._unnormalize(
            norm_state["effector_target_to_target_translation"],
            EFFECTOR_TARGET_TO_TARGET_TRANSLATION_MIN,
            EFFECTOR_TARGET_TO_TARGET_TRANSLATION_MAX,
        )
        target_translation = (
            effector_target_to_target_translation + effector_target_translation
        )
        ori_cos_sin = cls._unnormalize(
            norm_state["target_orientation_cos_sin"],
            TARGET_ORIENTATION_COS_SIN_MIN,
            TARGET_ORIENTATION_COS_SIN_MAX,
        )
        target_orientation = np.array(
            [math.atan2(ori_cos_sin[1], ori_cos_sin[0])], np.float32
        )

        return collections.OrderedDict(
            block_translation=block_translation,
            block_orientation=block_orientation,
            effector_translation=effector_translation,
            effector_target_translation=effector_target_translation,
            target_translation=target_translation,
            target_orientation=target_orientation,
        )

    def get_pybullet_state(self):
        return self._env.get_pybullet_state()

    def set_pybullet_state(self, state):
        return self._env.set_pybullet_state(state)

    @property
    def pybullet_client(self):
        return self._env.pybullet_client

    def calc_camera_params(self, image_size):
        return self._env.calc_camera_params(image_size)

    def _compute_state(self):
        return self.calc_normalized_state(
            self._env._compute_state()
        )  # pylint: disable=protected-access


# Make sure we only register once to allow us to reload the module in colab for
# debugging.
if "BlockPush-v0" in registration.registry.env_specs:
    del registration.registry.env_specs["BlockInsert-v0"]
    del registration.registry.env_specs["BlockPush-v0"]
    del registration.registry.env_specs["BlockPushNormalized-v0"]
    del registration.registry.env_specs["BlockPushRgbNormalized-v0"]
    del registration.registry.env_specs["BlockReach-v0"]
    del registration.registry.env_specs["BlockReachNormalized-v0"]
    del registration.registry.env_specs["BlockReachRgbNormalized-v0"]
    del registration.registry.env_specs["SharedBlockInsert-v0"]
    del registration.registry.env_specs["SharedBlockPush-v0"]
    del registration.registry.env_specs["SharedBlockReach-v0"]

registration.register(
    id="BlockInsert-v0",
    entry_point=BlockPush,
    kwargs=dict(task=BlockTaskVariant.INSERT),
    max_episode_steps=50,
)
registration.register(id="BlockPush-v0", entry_point=BlockPush, max_episode_steps=100)
registration.register(
    id="BlockPushNormalized-v0",
    entry_point=BlockPushNormalized,
    kwargs=dict(task=BlockTaskVariant.PUSH_NORMALIZED),
    max_episode_steps=100,
)
registration.register(
    id="BlockPushRgb-v0",
    entry_point=BlockPush,
    max_episode_steps=100,
    kwargs=dict(image_size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
)
registration.register(
    id="BlockPushRgbNormalized-v0",
    entry_point=BlockPushNormalized,
    kwargs=dict(
        task=BlockTaskVariant.PUSH_NORMALIZED, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
    ),
    max_episode_steps=100,
)
registration.register(
    id="BlockReach-v0",
    entry_point=BlockPush,
    kwargs=dict(task=BlockTaskVariant.REACH),
    max_episode_steps=50,
)
registration.register(
    id="BlockReachRgb-v0",
    entry_point=BlockPush,
    max_episode_steps=100,
    kwargs=dict(task=BlockTaskVariant.REACH, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
)
registration.register(
    id="BlockReachNormalized-v0",
    entry_point=BlockPushNormalized,
    kwargs=dict(task=BlockTaskVariant.REACH_NORMALIZED),
    max_episode_steps=50,
)
registration.register(
    id="BlockReachRgbNormalized-v0",
    entry_point=BlockPushNormalized,
    kwargs=dict(
        task=BlockTaskVariant.REACH_NORMALIZED, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
    ),
    max_episode_steps=50,
)

registration.register(
    id="SharedBlockInsert-v0",
    entry_point=BlockPush,
    kwargs=dict(task=BlockTaskVariant.INSERT, shared_memory=True),
    max_episode_steps=50,
)
registration.register(
    id="SharedBlockPush-v0",
    entry_point=BlockPush,
    kwargs=dict(shared_memory=True),
    max_episode_steps=100,
)
registration.register(
    id="SharedBlockPushNormalized-v0",
    entry_point=BlockPushNormalized,
    kwargs=dict(task=BlockTaskVariant.PUSH_NORMALIZED, shared_memory=True),
    max_episode_steps=100,
)
registration.register(
    id="SharedBlockReach-v0",
    entry_point=BlockPush,
    kwargs=dict(task=BlockTaskVariant.REACH, shared_memory=True),
    max_episode_steps=50,
)
