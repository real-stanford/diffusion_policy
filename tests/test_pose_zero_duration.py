"""Regression checks for instantaneous waypoint commands.

Run with: python -m unittest discover -s tests -p 'test_pose_zero_duration.py'
"""

import unittest

import numpy as np
from diffusion_policy.common.pose_trajectory_interpolator import (
    PoseTrajectoryInterpolator,
)
from scipy.spatial.transform import Rotation


class TestZeroDurationWaypoint(unittest.TestCase):
    def setUp(self):
        self.poses = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                [2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.interp = PoseTrajectoryInterpolator([0.0, 1.0, 2.0], self.poses)

    def assertPoseEqual(self, actual, expected):
        np.testing.assert_allclose(actual[:3], expected[:3], atol=1e-12)
        delta = (
            Rotation.from_rotvec(actual[3:]) * Rotation.from_rotvec(expected[3:]).inv()
        )
        self.assertLess(delta.magnitude(), 1e-12)

    def test_stationary_drive_at_current_time(self):
        target = self.interp(1.0)
        result = self.interp.drive_to_waypoint(target, 1.0, 1.0, 0.5, 0.25)
        np.testing.assert_array_equal(result.times, [1.0])
        self.assertPoseEqual(result(1.0), target)
        self.assertPoseEqual(result(100.0), target)

    def test_stationary_drive_with_elapsed_deadline(self):
        target = self.interp(1.0)
        result = self.interp.drive_to_waypoint(target, 0.5, 1.0, 0.5, 0.25)
        np.testing.assert_array_equal(result.times, [1.0])
        self.assertPoseEqual(result(1.0), target)

    def test_instantaneous_drive_replaces_current_pose(self):
        target = np.array([3.0, -1.0, 2.0, 0.0, 0.0, 0.25])
        result = self.interp.drive_to_waypoint(target, 1.0, 1.0)
        np.testing.assert_array_equal(result.times, [1.0])
        self.assertPoseEqual(result(1.0), target)

    def test_schedule_existing_endpoint_preserves_waypoints(self):
        result = self.interp.schedule_waypoint(self.poses[-1], 2.0)
        np.testing.assert_array_equal(result.times, self.interp.times)
        np.testing.assert_allclose(
            result(np.linspace(0.0, 2.0, 9)),
            self.interp(np.linspace(0.0, 2.0, 9)),
            atol=1e-12,
        )

    def test_instantaneous_schedule_replaces_endpoint(self):
        target = np.array([3.0, 1.0, 2.0, 0.0, 0.0, 0.25])
        result = self.interp.schedule_waypoint(target, 2.0)
        np.testing.assert_array_equal(result.times, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(result.poses[:-1], self.poses[:-1], atol=1e-12)
        self.assertPoseEqual(result(2.0), target)
        np.testing.assert_allclose(self.interp.poses, self.poses, atol=1e-12)

    def test_single_waypoint_schedule_replacement(self):
        interp = PoseTrajectoryInterpolator([1.0], [self.poses[0]])
        result = interp.schedule_waypoint(self.poses[-1], 1.0)
        np.testing.assert_array_equal(result.times, [1.0])
        self.assertPoseEqual(result(1.0), self.poses[-1])

    def test_finite_speed_drive_keeps_required_duration(self):
        target = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        result = self.interp.drive_to_waypoint(target, 1.0, 1.0, 0.5, 0.25)
        np.testing.assert_array_equal(result.times, [1.0, 5.0])
        self.assertPoseEqual(result(1.0), self.poses[1])
        self.assertPoseEqual(result(5.0), target)

    def test_finite_speed_schedule_extends_endpoint(self):
        target = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 1.5])
        result = self.interp.schedule_waypoint(target, 2.0, 0.5, 0.25)
        np.testing.assert_array_equal(result.times, [0.0, 1.0, 2.0, 4.0])
        self.assertPoseEqual(result(2.0), self.poses[-1])
        self.assertPoseEqual(result(4.0), target)

    def test_future_stationary_command_keeps_requested_time(self):
        target = self.interp(1.0)
        result = self.interp.drive_to_waypoint(target, 2.0, 1.0, 0.5, 0.25)
        np.testing.assert_array_equal(result.times, [1.0, 2.0])
        self.assertPoseEqual(result(2.0), target)

    def test_elapsed_schedule_keeps_existing_trajectory(self):
        result = self.interp.schedule_waypoint(self.poses[0], 0.5, curr_time=1.0)
        self.assertIs(result, self.interp)

    def test_controller_style_schedule_replaces_future_target(self):
        target = np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.75])
        result = self.interp.schedule_waypoint(
            target, 2.0, 1.0, 1.0, curr_time=1.0, last_waypoint_time=2.0
        )
        # The RTDE caller passes curr_time and last_waypoint_time, so a
        # replacement at the previous target's timestamp remains a segment.
        np.testing.assert_array_equal(result.times, [1.0, 2.0])
        self.assertPoseEqual(result(1.0), self.poses[1])
        self.assertPoseEqual(result(2.0), target)

    def test_finite_speed_controller_style_drive_keeps_delay(self):
        target = self.interp(1.0)
        result = self.interp.drive_to_waypoint(target, 1.01, 1.0, 0.5, 0.25)
        np.testing.assert_array_equal(result.times, [1.0, 1.01])
        self.assertPoseEqual(result(1.01), target)


if __name__ == "__main__":
    unittest.main()
