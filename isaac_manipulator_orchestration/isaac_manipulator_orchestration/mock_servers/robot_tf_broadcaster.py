#!/usr/bin/env python3

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from geometry_msgs.msg import TransformStamped
import rclpy
from rclpy.node import Node
import tf2_ros


class RobotTFBroadcaster(Node):
    """A TF broadcaster that provides robot kinematic transforms for mock scenarios."""

    def __init__(self):
        super().__init__('robot_tf_broadcaster')

        # Create static transform broadcaster
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # Publish static transforms on startup
        self._publish_static_transforms()

    def _publish_static_transforms(self):
        """Publish static transforms for robot kinematics."""
        transforms = []

        # Transform from base_link to gripper_frame
        # This represents the robot's end effector position in mock scenarios
        # In real scenarios, this would come from robot_state_publisher with the URDF
        gripper_transform = TransformStamped()
        gripper_transform.header.stamp = self.get_clock().now().to_msg()
        gripper_transform.header.frame_id = 'base_link'
        gripper_transform.child_frame_id = 'gripper_frame'

        # Use the home pose from blackboard params:
        # [-0.25, 0.45, 0.50, -0.677772, 0.734752, 0.020993, 0.017994]
        # Format: [x, y, z, qx, qy, qz, qw]
        gripper_transform.transform.translation.x = -0.25
        gripper_transform.transform.translation.y = 0.45
        gripper_transform.transform.translation.z = 0.50

        # Orientation from home pose quaternion
        gripper_transform.transform.rotation.x = -0.677772
        gripper_transform.transform.rotation.y = 0.734752
        gripper_transform.transform.rotation.z = 0.020993
        gripper_transform.transform.rotation.w = 0.017994

        transforms.append(gripper_transform)

        # Transform from gripper_frame to grasp_frame
        # This is defined in the robot URDF (ur_robotiq_gripper.urdf.xacro)
        # Joint "grasp_joint" has origin xyz="0 0 0.20" from gripper_frame to grasp_frame
        grasp_transform = TransformStamped()
        grasp_transform.header.stamp = self.get_clock().now().to_msg()
        grasp_transform.header.frame_id = 'gripper_frame'
        grasp_transform.child_frame_id = 'grasp_frame'

        # Values from URDF: xyz="0 0 0.20" rpy="0 0 0"
        grasp_transform.transform.translation.x = 0.0
        grasp_transform.transform.translation.y = 0.0
        grasp_transform.transform.translation.z = 0.20  # 20cm as defined in URDF

        # No rotation as defined in URDF
        grasp_transform.transform.rotation.x = 0.0
        grasp_transform.transform.rotation.y = 0.0
        grasp_transform.transform.rotation.z = 0.0
        grasp_transform.transform.rotation.w = 1.0

        transforms.append(grasp_transform)

        # Publish all transforms
        self.tf_static_broadcaster.sendTransform(transforms)

        self.get_logger().info(
            'Published robot kinematic transforms: base_link -> gripper_frame -> grasp_frame')


def main(args=None):
    rclpy.init(args=args)

    # Create the broadcaster node
    robot_broadcaster = RobotTFBroadcaster()

    try:
        robot_broadcaster.get_logger().info(
            'Starting robot TF broadcaster. Press Ctrl+C to exit.')
        rclpy.spin(robot_broadcaster)
    except KeyboardInterrupt:
        robot_broadcaster.get_logger().info('Robot TF broadcaster stopped by user')
    finally:
        # Clean up
        robot_broadcaster.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
