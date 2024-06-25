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

"""Reach oracle."""
import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

# Only used for debug visualization.
import pybullet  # pylint: disable=unused-import


class ReachOracle(py_policy.PyPolicy):
    """Oracle for moving to a specific spot relative to the block and target."""

    def __init__(self, env, block_pushing_oracles_action_std=0.0):
        super(ReachOracle, self).__init__(env.time_step_spec(), env.action_spec())
        self._env = env
        self._np_random_state = np.random.RandomState(0)
        self._block_pushing_oracles_action_std = block_pushing_oracles_action_std

    def _action(self, time_step, policy_state):

        # Specifying this as velocity makes it independent of control frequency.
        max_step_velocity = 0.2

        xy_ee = time_step.observation["effector_target_translation"]

        # This should be observable from block and target translation,
        # but re-using the computation from the env so that it's only done once, and
        # used for reward / completion computation.
        xy_pre_block = self._env.reach_target_translation

        xy_delta = xy_pre_block - xy_ee

        if self._block_pushing_oracles_action_std != 0.0:
            xy_delta += (
                self._np_random_state.randn(2) * self._block_pushing_oracles_action_std
            )

        max_step_distance = max_step_velocity * (1 / self._env.get_control_frequency())
        length = np.linalg.norm(xy_delta)
        if length > max_step_distance:
            xy_direction = xy_delta / length
            xy_delta = xy_direction * max_step_distance

        return policy_step.PolicyStep(action=np.asarray(xy_delta, dtype=np.float32))
