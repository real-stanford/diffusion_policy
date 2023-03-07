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

"""A simple 6DOF pose container.
"""

import dataclasses
import numpy as np
from scipy.spatial import transform


class NoCopyAsDict(object):
    """Base class for dataclasses. Avoids a copy in the asdict() call."""

    def asdict(self):
        """Replacement for dataclasses.asdict.

        TF Dataset does not handle dataclasses.asdict, which uses copy.deepcopy when
        setting values in the output dict. This causes issues with tf.Dataset.
        Instead, shallow copy contents.

        Returns:
          dict containing contents of dataclass.
        """
        return {k.name: getattr(self, k.name) for k in dataclasses.fields(self)}


@dataclasses.dataclass
class Pose3d(NoCopyAsDict):
    """Simple container for translation and rotation."""

    rotation: transform.Rotation
    translation: np.ndarray

    @property
    def vec7(self):
        return np.concatenate([self.translation, self.rotation.as_quat()])

    def serialize(self):
        return {
            "rotation": self.rotation.as_quat().tolist(),
            "translation": self.translation.tolist(),
        }

    @staticmethod
    def deserialize(data):
        return Pose3d(
            rotation=transform.Rotation.from_quat(data["rotation"]),
            translation=np.array(data["translation"]),
        )

    def __eq__(self, other):
        return np.array_equal(
            self.rotation.as_quat(), other.rotation.as_quat()
        ) and np.array_equal(self.translation, other.translation)

    def __ne__(self, other):
        return not self.__eq__(other)
