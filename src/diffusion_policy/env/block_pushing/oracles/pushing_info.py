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

"""Dataclass holding info needed for pushing oracles."""
import dataclasses
from typing import Any


@dataclasses.dataclass
class PushingInfo:
    """Holds onto info necessary for pushing state machine."""

    xy_block: Any = None
    xy_ee: Any = None
    xy_pre_block: Any = None
    xy_delta_to_nexttoblock: Any = None
    xy_delta_to_touchingblock: Any = None
    xy_dir_block_to_ee: Any = None
    theta_threshold_to_orient: Any = None
    theta_threshold_flat_enough: Any = None
    theta_error: Any = None
    obstacle_poses: Any = None
    distance_to_target: Any = None
