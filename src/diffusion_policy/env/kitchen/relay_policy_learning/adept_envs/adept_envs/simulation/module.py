#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for caching Python modules related to simulation."""

import sys

_MUJOCO_PY_MODULE = None

_DM_MUJOCO_MODULE = None
_DM_VIEWER_MODULE = None
_DM_RENDER_MODULE = None

_GLFW_MODULE = None


def get_mujoco_py():
    """Returns the mujoco_py module."""
    global _MUJOCO_PY_MODULE
    if _MUJOCO_PY_MODULE:
        return _MUJOCO_PY_MODULE
    try:
        import mujoco_py
        # Override the warning function.
        from mujoco_py.builder import cymj
        cymj.set_warning_callback(_mj_warning_fn)
    except ImportError:
        print(
            'Failed to import mujoco_py. Ensure that mujoco_py (using MuJoCo '
            'v1.50) is installed.',
            file=sys.stderr)
        sys.exit(1)
    _MUJOCO_PY_MODULE = mujoco_py
    return mujoco_py


def get_mujoco_py_mjlib():
    """Returns the mujoco_py mjlib module."""

    class MjlibDelegate:
        """Wrapper that forwards mjlib calls."""

        def __init__(self, lib):
            self._lib = lib

        def __getattr__(self, name: str):
            if name.startswith('mj'):
                return getattr(self._lib, '_' + name)
            raise AttributeError(name)

    return MjlibDelegate(get_mujoco_py().cymj)


def get_dm_mujoco():
    """Returns the DM Control mujoco module."""
    global _DM_MUJOCO_MODULE
    if _DM_MUJOCO_MODULE:
        return _DM_MUJOCO_MODULE
    try:
        from dm_control import mujoco
    except ImportError:
        print(
            'Failed to import dm_control.mujoco. Ensure that dm_control (using '
            'MuJoCo v2.00) is installed.',
            file=sys.stderr)
        sys.exit(1)
    _DM_MUJOCO_MODULE = mujoco
    return mujoco


def get_dm_viewer():
    """Returns the DM Control viewer module."""
    global _DM_VIEWER_MODULE
    if _DM_VIEWER_MODULE:
        return _DM_VIEWER_MODULE
    try:
        from dm_control import viewer
    except ImportError:
        print(
            'Failed to import dm_control.viewer. Ensure that dm_control (using '
            'MuJoCo v2.00) is installed.',
            file=sys.stderr)
        sys.exit(1)
    _DM_VIEWER_MODULE = viewer
    return viewer


def get_dm_render():
    """Returns the DM Control render module."""
    global _DM_RENDER_MODULE
    if _DM_RENDER_MODULE:
        return _DM_RENDER_MODULE
    try:
        try:
            from dm_control import _render
            render = _render
        except ImportError:
            print('Warning: DM Control is out of date.')
            from dm_control import render
    except ImportError:
        print(
            'Failed to import dm_control.render. Ensure that dm_control (using '
            'MuJoCo v2.00) is installed.',
            file=sys.stderr)
        sys.exit(1)
    _DM_RENDER_MODULE = render
    return render


def _mj_warning_fn(warn_data: bytes):
    """Warning function override for mujoco_py."""
    print('WARNING: Mujoco simulation is unstable (has NaNs): {}'.format(
        warn_data.decode()))
