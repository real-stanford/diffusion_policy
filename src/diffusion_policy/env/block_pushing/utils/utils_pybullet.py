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

"""Assortment of utilities to interact with bullet within g3."""
import dataclasses
import datetime
import getpass
import gzip
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from absl import logging
from diffusion_policy.env.block_pushing.utils.pose3d import Pose3d
import numpy as np
from scipy.spatial import transform
import six


import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bullet_client

Vec3 = Tuple[float, float, float]
Vec4 = Tuple[float, float, float, float]
PYBULLET_STATE_VERSION = 2  # Basic versioning of serialized pybullet state.


# Note about rotation_to_matrix and matrix_to_rotation below:
# The abstractions below allow us to use older versions of scipy.
def rotation_to_matrix(rotation):
    if hasattr(rotation, "as_dcm"):
        return rotation.as_dcm()
    else:
        assert hasattr(rotation, "as_matrix")
        return rotation.as_matrix()


def matrix_to_rotation(matrix):
    if hasattr(transform.Rotation, "from_dcm"):
        return transform.Rotation.from_dcm(matrix)
    else:
        assert hasattr(transform.Rotation, "from_matrix")
        return transform.Rotation.from_matrix(matrix)


def load_urdf(pybullet_client, file_path, *args, **kwargs):
    """Loads the given URDF filepath."""

    # Handles most general file open case.
    try:
        if os.path.exists(file_path):
            return pybullet_client.loadURDF(file_path, *args, **kwargs)
    except pybullet_client.error:
        pass

    try:
        import pathlib
        asset_path = str(pathlib.Path(__file__).parent.parent.joinpath('assets'))
        if file_path.startswith("third_party/py/envs/assets/"):
            pybullet_client.setAdditionalSearchPath(asset_path)
            file_path = file_path[len("third_party/py/envs/assets/") :]
        if file_path.startswith(
            "third_party/bullet/examples/pybullet/gym/pybullet_data/"
            ):
            pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
            file_path = file_path[55:]
        # logging.info("Loading URDF %s", file_path)
        return pybullet_client.loadURDF(file_path, *args, **kwargs)
    except pybullet.error:
        raise FileNotFoundError("Cannot load the URDF file {}".format(file_path))


def add_visual_sphere(client, center=(0, 0, 0), radius=0.1, rgba=(0.5, 0.5, 0.5, 0.5)):
    """Add a sphere to bullet scene (visual only, no physics).

    Args:
      client: pybullet client (or pybullet library handle).
      center: Center of sphere.
      radius: Sphere radius.
      rgba: rgba color of sphere.

    Returns:
      Unique integer bullet id of constructed object.
    """
    vis_obj_id = client.createVisualShape(
        client.GEOM_SPHERE, radius=radius, rgbaColor=rgba
    )
    obj_id = client.createMultiBody(
        baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_obj_id, basePosition=center
    )
    return obj_id


def pybullet_mat_to_numpy_4x4(pybullet_matrix):
    assert len(pybullet_matrix) == 16, "pybullet matrix should be len 16"
    return np.transpose(np.reshape(np.array(pybullet_matrix, dtype=np.float64), (4, 4)))


def decompose_view_matrix(pybullet_view_matrix):
    """Decompose view matrix into pos + quat format (assumes mat is rigid!)."""
    # It would be MUCH better to use something from bullet, however pybullet does
    # not expose all of the linear algebra library.
    mat = pybullet_mat_to_numpy_4x4(pybullet_view_matrix)

    # View matrix is now:
    # | R_11 R_12 R_13 t_1 |
    # | R_21 R_22 R_23 t_2 |
    # | R_31 R_32 R_33 t_3 |
    # |    0    0    0   1 |

    # R is the inverse eye to target at orientation, and t is R * eye.
    mat_view_to_world = np.linalg.inv(mat)

    # mat_view_to_world is the view to world transform, therefore the translation
    # component of this matrix is simply the world space position (since mat *
    # (0, 0, 0, 1)) is just copying the right column.
    world_xyz_view = np.copy(mat_view_to_world[0:3, 3])

    mat_view_to_world[0:3, 3] = 0  # Zero out the position change.
    world_quat_view = matrix_to_rotation(mat_view_to_world).as_quat()

    return world_xyz_view, world_quat_view


def world_obj_to_view(world_xyz_obj, world_quat_obj, camera_view, client):
    """Transform object into view space."""
    world_xyz_view, world_quat_view = decompose_view_matrix(camera_view)
    view_xyz_world, view_quat_world = client.invertTransform(
        world_xyz_view, world_quat_view
    )
    view_xyz_obj, view_quat_obj = client.multiplyTransforms(
        view_xyz_world, view_quat_world, world_xyz_obj, world_quat_obj
    )

    return view_xyz_obj, view_quat_obj


def image_xy_to_view_ray(xy, cam_width, cam_height, proj_mat_inv):
    """Calculate view-space ray from pixel location."""
    # Recall (from http://www.songho.ca/opengl/gl_projectionmatrix.html):
    # xyzw_clip = M_proj * xyzw_eye, and
    # xyz_ndc = xyzw_clip[0:3] / xwzw_clip[3].
    xyz_ndc = np.array(
        [2.0 * xy[0] / cam_width - 1.0, -(2.0 * xy[1] / cam_height - 1.0), 0]
    )  # in [-1, 1]
    xyzw_clip = np.concatenate([xyz_ndc, [1]])
    xyzw_eye = proj_mat_inv @ xyzw_clip
    origin = np.zeros(3)
    vec = xyzw_eye[:3] / max(np.linalg.norm(xyzw_eye[:3]), 1e-6)
    return origin, vec


def view_ray_to_world_ray(origin, vec, view_mat_inv):
    """Transform view-space ray into world space."""
    origin = view_mat_inv @ np.concatenate([origin, [1]])
    vec = view_mat_inv @ np.concatenate([vec, [0]])

    return origin[:3], vec[:3]


def ray_to_plane_test(ray_origin, ray_vec, plane_origin, plane_normal):
    """Perform a ray-plane intersection test."""
    ln = np.dot(plane_normal, ray_vec)
    if abs(ln) < np.finfo(np.float32).eps:
        return None

    # Solve for the intersection fraction t.
    t = np.dot(plane_normal, plane_origin - ray_origin) / ln
    if t >= 0:
        return ray_origin + ray_vec * t
    else:
        return None


def get_workspace(env):
    (
        workspace_origin,
        workspace_quat,
    ) = env.pybullet_client.getBasePositionAndOrientation(env.workspace_uid)
    workspace_normal = rotation_to_matrix(transform.Rotation.from_quat(workspace_quat))[
        2, 0:3
    ]

    return workspace_origin, workspace_normal


def reset_camera_pose(env, view_type):
    """Reset camera pose to canonical frame."""
    p = env.pybullet_client

    if view_type == "POLICY":
        camera_info = p.getDebugVisualizerCamera()
        image_size = (camera_info[0], camera_info[1])

        viewm, _, front_position, lookat, _ = env.calc_camera_params(image_size)

        euler = matrix_to_rotation(pybullet_mat_to_numpy_4x4(viewm)[0:3, 0:3]).as_euler(
            "xyz", degrees=False
        )
        pitch = euler[1]
        yaw = -euler[2]
        # The distance is a bit far away (the GL view has higher FOV).
        distance = np.linalg.norm(front_position - lookat) * 0.6
    elif view_type == "TOP_DOWN":
        workspace_origin, _ = get_workspace(env)
        distance = 0.5
        lookat = workspace_origin
        yaw = np.pi / 2
        # Note: pi/2 pitch results in gimble lock and pybullet doesn't support it.
        pitch = -(np.pi / 2 - 1e-5)
    else:
        raise ValueError("unsupported view_type %s" % view_type)
    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=360 * yaw / (2.0 * np.pi),
        cameraPitch=360 * pitch / (2.0 * np.pi),
        cameraTargetPosition=lookat,
    )


def _lists_to_tuple(obj):
    if isinstance(obj, list):
        return tuple([_lists_to_tuple(v) for v in obj])
    else:
        return obj


@dataclasses.dataclass
class ObjState:
    """A container for storing pybullet object state."""

    obj_id: int

    # base_pose: (xyz, quat).
    base_pose: Tuple[Vec3, Vec4]
    # base_vel: (vel, ang_vel).
    base_vel: Tuple[Vec3, Vec3]
    joint_info: Any
    joint_state: Any

    @staticmethod
    def get_bullet_state(client, obj_id):
        """Read Pybullet internal state."""
        base_pose = client.getBasePositionAndOrientation(obj_id)
        base_vel = client.getBaseVelocity(obj_id)

        joint_info = []
        joint_state = []
        for i in range(client.getNumJoints(obj_id)):
            joint_state.append(client.getJointState(obj_id, i))
            joint_info.append(ObjState._get_joint_info(client, obj_id, i))

        return ObjState(
            obj_id=obj_id,
            base_pose=base_pose,
            base_vel=base_vel,
            joint_info=tuple(joint_info),
            joint_state=tuple(joint_state),
        )

    @staticmethod
    def _get_joint_info(client, obj_id, joint_index):
        ji = client.getJointInfo(obj_id, joint_index)
        return tuple([v if not isinstance(v, bytes) else v.decode("utf-8") for v in ji])

    def set_bullet_state(self, client, obj_id):
        """Hard set the current bullet state."""
        xyz, quat = self.base_pose
        client.resetBasePositionAndOrientation(obj_id, xyz, quat)
        vel, ang_vel = self.base_vel
        client.resetBaseVelocity(obj_id, vel, ang_vel)

        njoints = client.getNumJoints(obj_id)
        if njoints != len(self.joint_info) or njoints != len(self.joint_state):
            raise ValueError("Incorrect number of joint info state pairs.")

        for i, (joint_info, joint_state) in enumerate(
            zip(self.joint_info, self.joint_state)
        ):
            joint_index = joint_info[0]
            if joint_index != i:
                raise ValueError("Joint index mismatch.")

            # Check that the current joint we're trying to restore state for has the
            # same info as the state joint.
            cur_joint_info = ObjState._get_joint_info(client, obj_id, joint_index)
            if cur_joint_info != joint_info:
                raise ValueError(
                    "joint_info mismatch %s vs %s (expected)"
                    % (str(cur_joint_info), str(joint_info))
                )
            joint_position = joint_state[0]
            joint_velocity = joint_state[1]
            client.resetJointState(
                obj_id, i, targetValue=joint_position, targetVelocity=joint_velocity
            )

    def serialize(self):
        return {
            "obj_id": self.obj_id,
            "base_pose": self.base_pose,
            "base_vel": self.base_vel,
            "joint_info": self.joint_info,
            "joint_state": self.joint_state,
        }

    @staticmethod
    def deserialize(data):
        return ObjState(
            obj_id=_lists_to_tuple(data["obj_id"]),
            base_pose=_lists_to_tuple(data["base_pose"]),
            base_vel=_lists_to_tuple(data["base_vel"]),
            joint_info=_lists_to_tuple(data["joint_info"]),
            joint_state=_lists_to_tuple(data["joint_state"]),
        )


@dataclasses.dataclass
class XarmState(ObjState):
    """A container for storing pybullet robot state."""

    # The set point of the robot's controller.
    target_effector_pose: Pose3d
    goal_translation: Optional[Vec3]

    @staticmethod
    def get_bullet_state(client, obj_id, target_effector_pose, goal_translation):
        if goal_translation is not None:
            goal_translation = tuple(goal_translation.tolist())
        return XarmState(
            **dataclasses.asdict(ObjState.get_bullet_state(client, obj_id)),
            target_effector_pose=target_effector_pose,
            goal_translation=goal_translation
        )

    def serialize(self):
        data = ObjState.serialize(self)
        data["target_effector_pose"] = self.target_effector_pose.serialize()
        if self.goal_translation is not None:
            data["goal_translation"] = self.goal_translation
        else:
            data["goal_translation"] = []
        return data

    @staticmethod
    def deserialize(data):
        goal_translation = (
            None
            if not data["goal_translation"]
            else _lists_to_tuple(data["goal_translation"])
        )
        return XarmState(
            obj_id=data["obj_id"],
            base_pose=_lists_to_tuple(data["base_pose"]),
            base_vel=_lists_to_tuple(data["base_vel"]),
            joint_info=_lists_to_tuple(data["joint_info"]),
            joint_state=_lists_to_tuple(data["joint_state"]),
            goal_translation=goal_translation,
            target_effector_pose=Pose3d.deserialize(data["target_effector_pose"]),
        )


def _serialize_pybullet_state(pybullet_state):
    """Convert data to POD types."""
    if isinstance(pybullet_state, list):
        return [_serialize_pybullet_state(entry) for entry in pybullet_state]
    elif isinstance(pybullet_state, dict):
        assert "_serialized_obj_name" not in pybullet_state
        return {
            key: _serialize_pybullet_state(value)
            for key, value in pybullet_state.items()
        }
    elif isinstance(pybullet_state, (XarmState, ObjState)):
        return {
            "_serialized_obj_name": type(pybullet_state).__name__,
            "_serialized_data": pybullet_state.serialize(),
        }
    elif isinstance(pybullet_state, int):
        return pybullet_state
    else:
        raise ValueError(
            "Unhandled type for object %s, type %s"
            % (str(pybullet_state), type(pybullet_state))
        )


def _deserialize_pybullet_state(state):
    """Parse data from POD types."""
    if isinstance(state, list):
        return [_deserialize_pybullet_state(item) for item in state]
    elif isinstance(state, dict):
        if "_serialized_obj_name" in state:
            if state["_serialized_obj_name"] == XarmState.__name__:
                return XarmState.deserialize(state["_serialized_data"])
            elif state["_serialized_obj_name"] == ObjState.__name__:
                return ObjState.deserialize(state["_serialized_data"])
            else:
                raise ValueError("Unsupported: %s" % state["_serialized_obj_name"])
        else:
            return {
                key: _deserialize_pybullet_state(value) for key, value in state.items()
            }
    elif isinstance(state, int):
        return state
    else:
        raise ValueError("Unhandled type for object %s" % str(state))


def write_pybullet_state(filename, pybullet_state, task, actions=None):
    """Serialize pybullet state to json file."""
    import torch
    data = {
        "pybullet_state": _serialize_pybullet_state(pybullet_state),
        "state_version": PYBULLET_STATE_VERSION,
        "ts_ms": int(time.mktime(datetime.datetime.now().timetuple())) * 1000,
        "user": getpass.getuser(),
        "task": task,
        "actions": actions if actions is not None else [],
    }
    torch.save(data, filename)


def read_pybullet_state(filename):
    """Deserialize pybullet state from json file."""
    import torch
    data = torch.load(filename)

    assert isinstance(data, dict)

    if data["state_version"] != PYBULLET_STATE_VERSION:
        raise ValueError(
            "incompatible state data (version %d, expected %d)"
            % (data["state_version"], PYBULLET_STATE_VERSION)
        )

    data["pybullet_state"] = _deserialize_pybullet_state(data["pybullet_state"])
    return data
