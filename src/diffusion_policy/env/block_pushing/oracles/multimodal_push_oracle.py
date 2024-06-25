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

"""Oracle for multimodal pushing task."""
import diffusion_policy.env.block_pushing.oracles.oriented_push_oracle as oriented_push_oracle_module
import numpy as np
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

# Only used for debug visualization.
import pybullet  # pylint: disable=unused-import


class MultimodalOrientedPushOracle(oriented_push_oracle_module.OrientedPushOracle):
    """Oracle for multimodal pushing task."""

    def __init__(self, env, goal_dist_tolerance=0.04, action_noise_std=0.0):
        super(MultimodalOrientedPushOracle, self).__init__(env)
        self._goal_dist_tolerance = goal_dist_tolerance
        self._action_noise_std = action_noise_std

    def reset(self):
        self.origin = None
        self.first_preblock = None
        self.phase = "move_to_pre_block"

    def _get_move_to_preblock(self, xy_pre_block, xy_ee):
        max_step_velocity = 0.3
        # Go 5 cm away from the block, on the line between the block and target.
        xy_delta_to_preblock = xy_pre_block - xy_ee
        diff = np.linalg.norm(xy_delta_to_preblock)
        if diff < 0.001:
            self.phase = "move_to_block"
            if self.first_preblock is None:
                self.first_preblock = np.copy(xy_pre_block)
        xy_delta = xy_delta_to_preblock
        return xy_delta, max_step_velocity

    def _get_action_for_block_target(self, time_step, block="block", target="target"):
        # Specifying this as velocity makes it independent of control frequency.
        max_step_velocity = 0.35

        info = self._get_action_info(time_step, block, target)

        if self.origin is None:
            self.origin = np.copy(info.xy_ee)

        if self.phase == "move_to_pre_block":
            xy_delta, max_step_velocity = self._get_move_to_preblock(
                info.xy_pre_block, info.xy_ee
            )

        if self.phase == "return_to_first_preblock":
            max_step_velocity = 0.3
            if self.first_preblock is None:
                self.first_preblock = self.origin
            # Return to the first preblock.
            xy_delta_to_origin = self.first_preblock - info.xy_ee
            diff = np.linalg.norm(xy_delta_to_origin)
            if diff < 0.001:
                self.phase = "return_to_origin"
            xy_delta = xy_delta_to_origin

        if self.phase == "return_to_origin":
            max_step_velocity = 0.3
            # Go 5 cm away from the block, on the line between the block and target.
            xy_delta_to_origin = self.origin - info.xy_ee
            diff = np.linalg.norm(xy_delta_to_origin)
            if diff < 0.001:
                self.phase = "move_to_pre_block"
            xy_delta = xy_delta_to_origin

        if self.phase == "move_to_block":
            xy_delta = self._get_move_to_block(
                info.xy_delta_to_nexttoblock,
                info.theta_threshold_to_orient,
                info.theta_error,
            )

        if self.phase == "push_block":
            xy_delta = self._get_push_block(
                info.theta_error,
                info.theta_threshold_to_orient,
                info.xy_delta_to_touchingblock,
            )

        orient_circle_diameter = 0.025

        if self.phase == "orient_block_left" or self.phase == "orient_block_right":
            max_step_velocity = 0.15

        if self.phase == "orient_block_left":
            xy_delta = self._get_orient_block_left(
                info.xy_dir_block_to_ee,
                orient_circle_diameter,
                info.xy_block,
                info.xy_ee,
                info.theta_error,
                info.theta_threshold_flat_enough,
            )

        if self.phase == "orient_block_right":
            xy_delta = self._get_orient_block_right(
                info.xy_dir_block_to_ee,
                orient_circle_diameter,
                info.xy_block,
                info.xy_ee,
                info.theta_error,
                info.theta_threshold_flat_enough,
            )

        if self._action_noise_std != 0.0:
            xy_delta += self._np_random_state.randn(2) * self._action_noise_std

        max_step_distance = max_step_velocity * (1 / self._env.get_control_frequency())
        length = np.linalg.norm(xy_delta)
        if length > max_step_distance:
            xy_direction = xy_delta / length
            xy_delta = xy_direction * max_step_distance
        return xy_delta

    def _choose_goal_order(self):
        """Chooses block->target order for multimodal pushing."""
        # Define all possible ((first_block, first_target),
        # (second_block, second_target)).
        possible_orders = [
            (("block", "target"), ("block2", "target2")),
            (("block", "target2"), ("block2", "target")),
            (("block2", "target"), ("block", "target2")),
            (("block2", "target2"), ("block", "target")),
        ]
        # import pdb; pdb.set_trace()
        # result = random.choice(possible_orders)
        result = possible_orders[self._env._rng.choice(len(possible_orders))]
        return result

    def _action(self, time_step, policy_state):
        if time_step.is_first():
            self.reset()
            (
                (self._first_block, self._first_target),
                (self._second_block, self._second_target),
            ) = self._choose_goal_order()
            self._current_block, self._current_target = (
                self._first_block,
                self._first_target,
            )
            self._has_switched = False

        def _block_target_dist(block, target):
            dist = np.linalg.norm(
                time_step.observation["%s_translation" % block]
                - time_step.observation["%s_translation" % target]
            )
            return dist

        if (
            _block_target_dist(self._first_block, self._first_target)
            < self._goal_dist_tolerance
            and not self._has_switched
        ):
            # If first block has been pushed to first target, switch to second block.
            self._current_block, self._current_target = (
                self._second_block,
                self._second_target,
            )
            self._has_switched = True
            self.phase = "return_to_first_preblock"

        xy_delta = self._get_action_for_block_target(
            time_step, block=self._current_block, target=self._current_target
        )

        return policy_step.PolicyStep(action=np.asarray(xy_delta, dtype=np.float32))
