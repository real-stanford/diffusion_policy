"""Regression coverage for the MuJoCo image dataset feature schema."""

import ast
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "diffusion_policy" / "dataset" / "mujoco_image_dataset.py"


class _RecordingNormalizer:
    recorded_data = None

    def fit(self, data, **kwargs):
        del kwargs
        type(self).recorded_data = data
        return self

    def __setitem__(self, key, value):
        del key, value


def _load_get_normalizer():
    """Extract the small method without importing optional dataset backends."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    dataset_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MujocoImageDataset"
    )
    method = next(
        node
        for node in dataset_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_normalizer"
    )
    namespace = {
        "np": np,
        "LinearNormalizer": _RecordingNormalizer,
        "get_image_range_normalizer": lambda: object(),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(MODULE_PATH), "exec"),
        namespace,
    )
    return namespace["get_normalizer"]


def test_normalizer_uses_gripper_width_for_agent_position():
    tcp = np.arange(14, dtype=np.float32).reshape(2, 7)
    gripper = np.array([[0.1], [0.2]], dtype=np.float32)
    action_tcp = np.zeros((2, 7), dtype=np.float32)
    action_gripper = np.zeros((2, 1), dtype=np.float32)

    dataset = type("DatasetStub", (), {})()
    dataset.replay_buffer = {
        "robot_0_tcp_xyz_wxyz": tcp,
        "robot_0_gripper_width": gripper,
        "action_0_tcp_xyz_wxyz": action_tcp,
        "action_0_gripper_width": action_gripper,
    }

    _RecordingNormalizer.recorded_data = None
    _load_get_normalizer()(dataset)

    agent_pos = _RecordingNormalizer.recorded_data["agent_pos"]
    expected = np.concatenate([tcp, gripper], axis=-1)
    np.testing.assert_array_equal(agent_pos, expected)
    assert agent_pos.shape == (2, 8)
