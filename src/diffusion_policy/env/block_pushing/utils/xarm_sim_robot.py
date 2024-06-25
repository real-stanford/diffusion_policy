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

"""XArm Robot Kinematics."""
from diffusion_policy.env.block_pushing.utils import utils_pybullet
from diffusion_policy.env.block_pushing.utils.pose3d import Pose3d
import numpy as np
from scipy.spatial import transform
import pybullet

XARM_URDF_PATH = (
    "third_party/bullet/examples/pybullet/gym/pybullet_data/" "xarm/xarm6_robot.urdf"
)
SUCTION_URDF_PATH = "third_party/py/envs/assets/suction/" "suction-head-long.urdf"
CYLINDER_URDF_PATH = "third_party/py/envs/assets/suction/" "cylinder.urdf"
CYLINDER_REAL_URDF_PATH = "third_party/py/envs/assets/suction/" "cylinder_real.urdf"
HOME_JOINT_POSITIONS = np.deg2rad([0, -20, -80, 0, 100, -30])


class XArmSimRobot:
    """A simulated PyBullet XArm robot, mostly for forward/inverse kinematics."""

    def __init__(
        self,
        pybullet_client,
        initial_joint_positions=HOME_JOINT_POSITIONS,
        end_effector="none",
        color="default",
    ):
        self._pybullet_client = pybullet_client
        self.initial_joint_positions = initial_joint_positions

        if color == "default":
            self.xarm = utils_pybullet.load_urdf(
                pybullet_client, XARM_URDF_PATH, [0, 0, 0]
            )
        else:
            raise ValueError("Unrecognized xarm color %s" % color)

        # Get revolute joints of robot (skip fixed joints).
        joints = []
        joint_indices = []
        for i in range(self._pybullet_client.getNumJoints(self.xarm)):
            joint_info = self._pybullet_client.getJointInfo(self.xarm, i)
            if joint_info[2] == pybullet.JOINT_REVOLUTE:
                joints.append(joint_info[0])
                joint_indices.append(i)
                # Note examples in pybullet do this, but it is not clear what the
                # benefits are.
                self._pybullet_client.changeDynamics(
                    self.xarm, i, linearDamping=0, angularDamping=0
                )

        self._n_joints = len(joints)
        self._joints = tuple(joints)
        self._joint_indices = tuple(joint_indices)

        # Move robot to home joint configuration
        self.reset_joints(self.initial_joint_positions)
        self.effector_link = 6

        if (
            end_effector == "suction"
            or end_effector == "cylinder"
            or end_effector == "cylinder_real"
        ):
            self.end_effector = self._setup_end_effector(end_effector)
        else:
            if end_effector != "none":
                raise ValueError('end_effector "%s" is not supported.' % end_effector)
            self.end_effector = None

    def _setup_end_effector(self, end_effector):
        """Adds a suction or cylinder end effector."""
        pose = self.forward_kinematics()
        if end_effector == "suction":
            body = utils_pybullet.load_urdf(
                self._pybullet_client,
                SUCTION_URDF_PATH,
                pose.translation,
                pose.rotation.as_quat(),
            )
        elif end_effector == "cylinder":
            body = utils_pybullet.load_urdf(
                self._pybullet_client,
                CYLINDER_URDF_PATH,
                pose.translation,
                pose.rotation.as_quat(),
            )
        elif end_effector == "cylinder_real":
            body = utils_pybullet.load_urdf(
                self._pybullet_client,
                CYLINDER_REAL_URDF_PATH,
                pose.translation,
                pose.rotation.as_quat(),
            )
        else:
            raise ValueError('end_effector "%s" is not supported.' % end_effector)

        constraint_id = self._pybullet_client.createConstraint(
            parentBodyUniqueId=self.xarm,
            parentLinkIndex=6,
            childBodyUniqueId=body,
            childLinkIndex=-1,
            jointType=pybullet.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0),
        )
        self._pybullet_client.changeConstraint(constraint_id, maxForce=50)

        return body

    def reset_joints(self, joint_values):
        """Sets the position of the Robot's joints.

        *Note*: This should only be used at the start while not running the
                simulation resetJointState overrides all physics simulation.

        Args:
          joint_values: Iterable with desired joint positions.
        """
        for i in range(self._n_joints):
            self._pybullet_client.resetJointState(
                self.xarm, self._joints[i], joint_values[i]
            )

    def get_joints_measured(self):
        joint_states = self._pybullet_client.getJointStates(
            self.xarm, self._joint_indices
        )
        joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        joint_torques = np.array([state[3] for state in joint_states])
        return joint_positions, joint_velocities, joint_torques

    def get_joint_positions(self):
        joint_states = self._pybullet_client.getJointStates(
            self.xarm, self._joint_indices
        )
        joint_positions = np.array([state[0] for state in joint_states])
        return joint_positions

    def forward_kinematics(self):
        """Forward kinematics."""
        effector_state = self._pybullet_client.getLinkState(
            self.xarm, self.effector_link
        )
        return Pose3d(
            translation=np.array(effector_state[0]),
            rotation=transform.Rotation.from_quat(effector_state[1]),
        )

    def inverse_kinematics(
        self, world_effector_pose, max_iterations=100, residual_threshold=1e-10
    ):
        """Inverse kinematics.

        Args:
          world_effector_pose: Target Pose3d for the robot's end effector.
          max_iterations: Refine the IK solution until the distance between target
            and actual end effector position is below this threshold, or the
            maxNumIterations is reached. Default is 20 iterations.
          residual_threshold: Refine the IK solution until the distance between
            target and actual end effector position is below this threshold, or the
            maxNumIterations is reached.

        Returns:
          Numpy array with required joint angles to reach the requested pose.
        """
        return np.array(
            self._pybullet_client.calculateInverseKinematics(
                self.xarm,
                self.effector_link,
                world_effector_pose.translation,
                world_effector_pose.rotation.as_quat(),  # as_quat returns xyzw.
                lowerLimits=[-17] * 6,
                upperLimits=[17] * 6,
                jointRanges=[17] * 6,
                restPoses=[0, 0] + self.get_joint_positions()[2:].tolist(),
                maxNumIterations=max_iterations,
                residualThreshold=residual_threshold,
            )
        )

    def set_target_effector_pose(self, world_effector_pose):
        target_joint_positions = self.inverse_kinematics(world_effector_pose)
        self.set_target_joint_positions(target_joint_positions)

    def set_target_joint_velocities(self, target_joint_velocities):
        self._pybullet_client.setJointMotorControlArray(
            self.xarm,
            self._joint_indices,
            pybullet.VELOCITY_CONTROL,
            targetVelocities=target_joint_velocities,
            forces=[5 * 240.0] * 6,
        )

    def set_target_joint_positions(self, target_joint_positions):
        self._pybullet_client.setJointMotorControlArray(
            self.xarm,
            self._joint_indices,
            pybullet.POSITION_CONTROL,
            targetPositions=target_joint_positions,
            forces=[5 * 240.0] * 6,
        )

    def set_alpha_transparency(self, alpha):
        visual_shape_data = self._pybullet_client.getVisualShapeData(self.xarm)

        for i in range(self._pybullet_client.getNumJoints(self.xarm)):
            object_id, link_index, _, _, _, _, _, rgba_color = visual_shape_data[i]
            assert object_id == self.xarm, "xarm id mismatch."
            assert link_index == i, "Link visual data was returned out of order."
            rgba_color = list(rgba_color[0:3]) + [alpha]
            self._pybullet_client.changeVisualShape(
                self.xarm, linkIndex=i, rgbaColor=rgba_color
            )
