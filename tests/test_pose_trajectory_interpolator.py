import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from tqdm import tqdm
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st
from diffusion_policy.common.pose_trajectory_interpolator import (
    rotation_distance, 
    pose_distance, 
    PoseTrajectoryInterpolator)


def test_rotation_distance():
    def rotation_distance_align(a: st.Rotation, b: st.Rotation) -> float:
        return st.Rotation.align_vectors(b.as_matrix().T, a.as_matrix().T)[0].magnitude()

    for i in range(10000):
        a = st.Rotation.from_euler('xyz', np.random.uniform(-7,7,size=3))
        b = st.Rotation.from_euler('xyz', np.random.uniform(-7,7,size=3))
        x = rotation_distance(a, b)
        y = rotation_distance_align(a, b)
        assert abs(x-y) < 1e-7

def test_pose_trajectory_interpolator():
    t = np.linspace(-1,5,100)
    interp = PoseTrajectoryInterpolator(
        [0,1,3],
        np.zeros((3,6))
    )
    times = interp.times
    poses = interp.poses

    trimmed_interp = interp.trim(-1,4)
    assert len(trimmed_interp.times) == 5
    trimmed_interp(t)

    trimmed_interp = interp.trim(-1,4)
    assert len(trimmed_interp.times) == 5
    trimmed_interp(t)

    trimmed_interp = interp.trim(0.5, 3.5)
    assert len(trimmed_interp.times) == 4
    trimmed_interp(t)

    trimmed_interp = interp.trim(0.5, 2.5)
    assert len(trimmed_interp.times) == 3
    trimmed_interp(t)

    trimmed_interp = interp.trim(0.5, 1.5)
    assert len(trimmed_interp.times) == 3
    trimmed_interp(t)

    trimmed_interp = interp.trim(1.2, 1.5)
    assert len(trimmed_interp.times) == 2
    trimmed_interp(t)

    trimmed_interp = interp.trim(1.3, 1.3)
    assert len(trimmed_interp.times) == 1
    trimmed_interp(t)

    # import pdb; pdb.set_trace()

def test_add_waypoint():
    # fuzz testing
    for i in tqdm(range(10000)):
        rng = np.random.default_rng(i)
        n_waypoints = rng.integers(1, 5)
        waypoint_times = np.sort(rng.uniform(0, 1, size=n_waypoints))
        last_waypoint_time = waypoint_times[-1]
        insert_time = rng.uniform(-0.1, 1.1)
        curr_time = rng.uniform(-0.1, 1.1)
        max_pos_speed = rng.poisson(3) + 1e-3
        max_rot_speed = rng.poisson(3) + 1e-3
        waypoint_poses = rng.normal(0, 3, size=(n_waypoints, 6))
        new_pose = rng.normal(0, 3, size=6)

        if rng.random() < 0.1:
            last_waypoint_time = None
            if rng.random() < 0.1:
                curr_time = None

        interp = PoseTrajectoryInterpolator(
            times=waypoint_times, 
            poses=waypoint_poses)
        new_interp = interp.add_waypoint(
            pose=new_pose,
            time=insert_time,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
            curr_time=curr_time,
            last_waypoint_time=last_waypoint_time
        )

def test_drive_to_waypoint():
    # fuzz testing
    for i in tqdm(range(10000)):
        rng = np.random.default_rng(i)
        n_waypoints = rng.integers(1, 5)
        waypoint_times = np.sort(rng.uniform(0, 1, size=n_waypoints))
        insert_time = rng.uniform(-0.1, 1.1)
        curr_time = rng.uniform(-0.1, 1.1)
        max_pos_speed = rng.poisson(3) + 1e-3
        max_rot_speed = rng.poisson(3) + 1e-3
        waypoint_poses = rng.normal(0, 3, size=(n_waypoints, 6))
        new_pose = rng.normal(0, 3, size=6)

        interp = PoseTrajectoryInterpolator(
            times=waypoint_times, 
            poses=waypoint_poses)
        new_interp = interp.drive_to_waypoint(
            pose=new_pose,
            time=insert_time,
            curr_time=curr_time,
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed
        )



if __name__ == '__main__':
    test_drive_to_waypoint()
