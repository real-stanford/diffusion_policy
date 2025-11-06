#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time
from threading import Lock

# ROS2 message imports
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, TwistStamped
from franka_msgs.msg import FrankaRobotState


class FrankaDiagnostics(Node):
    def __init__(self):
        super().__init__('franka_diagnostics')
        
        self.data_lock = Lock()
        
        # Counters for received messages
        self.joint_states_count = 0
        self.current_pose_count = 0
        self.desired_twist_count = 0
        self.measured_joint_states_count = 0
        self.robot_state_count = 0
        
        # Last received timestamps
        self.last_joint_states_time = 0
        self.last_current_pose_time = 0
        self.last_desired_twist_time = 0
        self.last_measured_joint_states_time = 0
        self.last_robot_state_time = 0
        
        # Setup QoS profiles
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Test different topics
        self.get_logger().info("Setting up topic subscribers...")
        
        try:
            self.joint_states_sub = self.create_subscription(
                JointState,
                '/joint_states',
                self.joint_states_callback,
                reliable_qos
            )
            self.get_logger().info("✓ Subscribed to /joint_states")
        except Exception as e:
            self.get_logger().error(f"✗ Failed to subscribe to /joint_states: {e}")
        
        try:
            self.current_pose_sub = self.create_subscription(
                PoseStamped,
                '/franka_robot_state_broadcaster/current_pose',
                self.current_pose_callback,
                best_effort_qos
            )
            self.get_logger().info("✓ Subscribed to /franka_robot_state_broadcaster/current_pose")
        except Exception as e:
            self.get_logger().error(f"✗ Failed to subscribe to current_pose: {e}")
        
        try:
            self.desired_twist_sub = self.create_subscription(
                TwistStamped,
                '/franka_robot_state_broadcaster/desired_end_effector_twist',
                self.desired_twist_callback,
                best_effort_qos
            )
            self.get_logger().info("✓ Subscribed to /franka_robot_state_broadcaster/desired_end_effector_twist")
        except Exception as e:
            self.get_logger().error(f"✗ Failed to subscribe to desired_twist: {e}")
        
        try:
            self.measured_joint_states_sub = self.create_subscription(
                JointState,
                '/franka_robot_state_broadcaster/measured_joint_states',
                self.measured_joint_states_callback,
                best_effort_qos
            )
            self.get_logger().info("✓ Subscribed to /franka_robot_state_broadcaster/measured_joint_states")
        except Exception as e:
            self.get_logger().error(f"✗ Failed to subscribe to measured_joint_states: {e}")
        
        try:
            self.robot_state_sub = self.create_subscription(
                FrankaRobotState,
                '/franka_robot_state_broadcaster/robot_state',
                self.robot_state_callback,
                best_effort_qos
            )
            self.get_logger().info("✓ Subscribed to /franka_robot_state_broadcaster/robot_state")
        except Exception as e:
            self.get_logger().error(f"✗ Failed to subscribe to robot_state: {e}")
        
        # Timer for periodic reporting
        self.report_timer = self.create_timer(2.0, self.report_status)
        
        self.get_logger().info("Franka Diagnostics started. Monitoring topics...")
        
    def joint_states_callback(self, msg):
        with self.data_lock:
            self.joint_states_count += 1
            self.last_joint_states_time = time.time()
            
    def current_pose_callback(self, msg):
        with self.data_lock:
            self.current_pose_count += 1
            self.last_current_pose_time = time.time()
            
    def desired_twist_callback(self, msg):
        with self.data_lock:
            self.desired_twist_count += 1
            self.last_desired_twist_time = time.time()
            
    def measured_joint_states_callback(self, msg):
        with self.data_lock:
            self.measured_joint_states_count += 1
            self.last_measured_joint_states_time = time.time()
            
    def robot_state_callback(self, msg):
        with self.data_lock:
            self.robot_state_count += 1
            self.last_robot_state_time = time.time()
    
    def report_status(self):
        current_time = time.time()
        
        with self.data_lock:
            self.get_logger().info("=== Topic Status Report ===")
            
            # Check /joint_states
            if self.joint_states_count > 0:
                age = current_time - self.last_joint_states_time
                self.get_logger().info(f"✓ /joint_states: {self.joint_states_count} msgs, last {age:.2f}s ago")
            else:
                self.get_logger().warn("✗ /joint_states: No messages received")
            
            # Check current_pose
            if self.current_pose_count > 0:
                age = current_time - self.last_current_pose_time
                self.get_logger().info(f"✓ current_pose: {self.current_pose_count} msgs, last {age:.2f}s ago")
            else:
                self.get_logger().warn("✗ current_pose: No messages received")
            
            # Check desired_twist
            if self.desired_twist_count > 0:
                age = current_time - self.last_desired_twist_time
                self.get_logger().info(f"✓ desired_twist: {self.desired_twist_count} msgs, last {age:.2f}s ago")
            else:
                self.get_logger().warn("✗ desired_twist: No messages received")
            
            # Check measured_joint_states
            if self.measured_joint_states_count > 0:
                age = current_time - self.last_measured_joint_states_time
                self.get_logger().info(f"✓ measured_joint_states: {self.measured_joint_states_count} msgs, last {age:.2f}s ago")
            else:
                self.get_logger().warn("✗ measured_joint_states: No messages received")
            
            # Check robot_state
            if self.robot_state_count > 0:
                age = current_time - self.last_robot_state_time
                self.get_logger().info(f"✓ robot_state: {self.robot_state_count} msgs, last {age:.2f}s ago")
            else:
                self.get_logger().warn("✗ robot_state: No messages received")
            
            self.get_logger().info("============================")


def list_available_topics():
    """List all available topics"""
    import subprocess
    import sys
    
    try:
        result = subprocess.run(['ros2', 'topic', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            topics = result.stdout.strip().split('\n')
            franka_topics = [t for t in topics if 'franka' in t.lower()]
            
            print("Available Franka-related topics:")
            for topic in franka_topics:
                print(f"  {topic}")
            
            print(f"\nAll topics ({len(topics)} total):")
            for topic in sorted(topics):
                print(f"  {topic}")
        else:
            print(f"Error listing topics: {result.stderr}")
    except Exception as e:
        print(f"Error running ros2 topic list: {e}")


def check_topic_info(topic_name):
    """Get detailed info about a specific topic"""
    import subprocess
    
    try:
        # Get topic type
        result = subprocess.run(['ros2', 'topic', 'info', topic_name], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"\nTopic info for {topic_name}:")
            print(result.stdout)
        
        # Get topic hz
        print(f"Checking publishing rate for {topic_name} (5 second sample)...")
        result = subprocess.run(['ros2', 'topic', 'hz', topic_name], 
                              capture_output=True, text=True, timeout=6)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"No messages received on {topic_name}")
            
    except subprocess.TimeoutExpired:
        print(f"Timeout checking {topic_name}")
    except Exception as e:
        print(f"Error checking topic {topic_name}: {e}")


def main(args=None):
    print("Franka Robot Diagnostics Tool")
    print("============================")
    
    # First, list available topics
    list_available_topics()
    
    # Check key topics
    key_topics = [
        '/joint_states',
        '/franka_robot_state_broadcaster/current_pose',
        '/franka_robot_state_broadcaster/desired_end_effector_twist',
        '/franka_robot_state_broadcaster/measured_joint_states'
    ]
    
    for topic in key_topics:
        check_topic_info(topic)
    
    # Start the diagnostic node
    rclpy.init(args=args)
    
    node = FrankaDiagnostics()
    
    try:
        print("\nStarting live topic monitoring...")
        print("Press Ctrl+C to stop")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down diagnostics...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()