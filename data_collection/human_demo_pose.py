import zarr
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np
import argparse

class PoseSubscriber(Node):
    def __init__(self,args):
        super().__init__('pose_subscriber')
        self.args = args

        # ROS topic subscribers (message_filters)
        self.pose_sub = Subscriber(self, PoseStamped, '/franka_robot_state_broadcaster/current_pose')
        self.joint_sub = Subscriber(self, JointState, '/franka_robot_state_broadcaster/measured_joint_states')
        self.twist_sub = Subscriber(self, TwistStamped, '/franka_robot_state_broadcaster/desired_end_effector_twist')

        # Synchronizer
        self.ts = ApproximateTimeSynchronizer(
            [self.pose_sub, self.joint_sub, self.twist_sub], 
            queue_size=10, 
            slop=0.05
        )
        self.ts.registerCallback(self.synced_callback)

        # Zarr setup
        zarr_file_path = os.path.expanduser('~/Documents/data_collection/data/replay_buffer.zarr')
        self.zarr_root = zarr.open(zarr_file_path, mode='a')
        self.data_group = self.zarr_root.require_group('data')
        self.meta_group = self.zarr_root.require_group('meta')

        if 'eef_pose' not in self.data_group:
            self.data_group.create_dataset('eef_pose', shape=(0, 7), maxshape=(None, 7), chunks=(100, 7), dtype='f4')
        if 'action' not in self.data_group:
            self.data_group.create_dataset('action', shape=(0, 7), maxshape=(None, 7), chunks=(100, 7), dtype='f4')
        if 'eef_velocity' not in self.data_group:
            self.data_group.create_dataset('eef_velocity', shape=(0, 6), maxshape=(None, 6), chunks=(100, 6), dtype='f4')
        if 'joint_pose' not in self.data_group:
            self.data_group.create_dataset('joint_pose', shape=(0, 7), maxshape=(None, 7), chunks=(100, 7), dtype='f4')
        if 'joint_velocity' not in self.data_group:
            self.data_group.create_dataset('joint_velocity', shape=(0, 7), maxshape=(None, 7), chunks=(100, 7), dtype='f4')
        if 'timestamp' not in self.data_group:
            self.data_group.create_dataset('timestamp',shape=(1,),maxshape=(None, 7), chunks=(100,))
        if 'episode_ends' not in self.meta_group:
            self.meta_group.create_dataset('episode_ends',shape=(0,),maxshape=(None,),chunks=(100,),dtype='f4')

        self.target_interval = 1.0 /self.args.fps
        self.last_recorded_time = self.get_clock().now()

    def synced_callback(self, pose_msg: PoseStamped, joint_msg: JointState, twist_msg: TwistStamped):
        current_time = self.get_clock().now()
        time_diff = (current_time - self.last_recorded_time).nanoseconds * 1e-9
        if time_diff < self.target_interval:
            return

        idx = self.data_group['eef_pose'].shape[0]
        self.data_group['timestamp'].resize((idx+1,))
        self.data_group['timestamp'][idx] = current_time.nanoseconds * 1e-9
        self.last_recorded_time = current_time

        pose_np = np.array([
            pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z,
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w
        ], dtype=np.float32)
        self.data_group['eef_pose'].resize((idx + 1, 7))
        self.data_group['eef_pose'][idx] = pose_np

        self.data_group['action'].resize((idx + 1, 7))
        self.data_group['action'][idx] = pose_np

        # Process and save eef_velocity
        vel_np = np.array([
            twist_msg.twist.linear.x,
            twist_msg.twist.linear.y,
            twist_msg.twist.linear.z,
            twist_msg.twist.angular.x,
            twist_msg.twist.angular.y,
            twist_msg.twist.angular.z,
        ], dtype=np.float32)
        self.data_group['eef_velocity'].resize((idx + 1, 6))
        self.data_group['eef_velocity'][idx] = vel_np

        # Process and save joint positions and velocities
        joint_pos = np.array(joint_msg.position, dtype=np.float32)
        joint_vel = np.array(joint_msg.velocity, dtype=np.float32)
        self.data_group['joint_pose'].resize((idx + 1, 7))
        self.data_group['joint_pose'][idx] = joint_pos
        self.data_group['joint_velocity'].resize((idx + 1, 7))
        self.data_group['joint_velocity'][idx] = joint_vel

        #self.get_logger().info(f"Synced data #{idx}: pose={pose_np}, vel={vel_np}, joints={joint_pos}")

def main():

    parser = argparse.ArgumentParser(description='Run both human video capture and pose logger')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--demo_idx', type=int, default=0, help='Demo index')
    args = parser.parse_args()

    rclpy.init()
    node = PoseSubscriber(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        end_idx = node.data_group['timestamp'].shape[0]
        ep_ds = node.meta_group['episode_ends']
        current_shape = ep_ds.shape[0]

        ep_ds.resize((current_shape + 1,))
        ep_ds[current_shape] = end_idx
        print(f"✅ Recorded episode {current_shape} ending at index {end_idx}")

        node.destroy_node()

        rclpy.shutdown()

if __name__ == '__main__':
    main()