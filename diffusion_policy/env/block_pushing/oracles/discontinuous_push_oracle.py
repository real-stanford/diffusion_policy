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

"""Pushes to first target, waits, then pushes to second target."""

import diffusion_policy.env.block_pushing.oracles.oriented_push_oracle as oriented_push_oracle_module
import numpy as np
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

# Only used for debug visualization.
import pybullet  # pylint: disable=unused-import


class DiscontinuousOrientedPushOracle(oriented_push_oracle_module.OrientedPushOracle):
    """Pushes to first target, waits, then pushes to second target."""

    def __init__(self, env, goal_tolerance=0.04, wait=0):
        super(DiscontinuousOrientedPushOracle, self).__init__(env)
        self._countdown = 0
        self._wait = wait
        self._goal_dist_tolerance = goal_tolerance

    def reset(self):
        self.phase = "move_to_pre_block"
        self._countdown = 0

    def _action(self, time_step, policy_state):
        if time_step.is_first():
            self.reset()
            # Move to first target first.
            self._current_target = "target"
            self._has_switched = False

        def _block_target_dist(block, target):
            dist = np.linalg.norm(
                time_step.observation["%s_translation" % block]
                - time_step.observation["%s_translation" % target]
            )
            return dist

        d1 = _block_target_dist("block", "target")
        if d1 < self._goal_dist_tolerance and not self._has_switched:
            self._countdown = self._wait
            # If first block has been pushed to first target, switch to second block.
            self._has_switched = True
            self._current_target = "target2"

        xy_delta = self._get_action_for_block_target(
            time_step, block="block", target=self._current_target
        )

        if self._countdown > 0:
            xy_delta = np.zeros_like(xy_delta)
            self._countdown -= 1

        return policy_step.PolicyStep(action=np.asarray(xy_delta, dtype=np.float32))
