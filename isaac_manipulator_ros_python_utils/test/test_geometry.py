# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

from geometry_msgs.msg import Point, Pose, Quaternion, Transform, Vector3
import isaac_manipulator_ros_python_utils.geometry as geometry_utils
import numpy as np


class TestGeometry(unittest.TestCase):
    """Test cases for geometry utility functions."""

    def test_s_to_ns(self):
        """Test seconds to nanoseconds conversion."""
        self.assertEqual(geometry_utils.s_to_ns(1.0), 1_000_000_000)
        self.assertEqual(geometry_utils.s_to_ns(0.5), 500_000_000)
        self.assertEqual(geometry_utils.s_to_ns(0.001), 1_000_000)

    def test_ns_to_s(self):
        """Test nanoseconds to seconds conversion."""
        self.assertEqual(geometry_utils.ns_to_s(1_000_000_000), 1.0)
        self.assertEqual(geometry_utils.ns_to_s(500_000_000), 0.5)
        self.assertEqual(geometry_utils.ns_to_s(1_000_000), 0.001)

    def test_deg_to_rad(self):
        """Test degrees to radians conversion."""
        self.assertAlmostEqual(geometry_utils.deg_to_rad(180), math.pi)
        self.assertAlmostEqual(geometry_utils.deg_to_rad(90), math.pi / 2)
        self.assertAlmostEqual(geometry_utils.deg_to_rad(360), 2 * math.pi)

    def test_rad_to_deg(self):
        """Test radians to degrees conversion."""
        self.assertAlmostEqual(geometry_utils.rad_to_deg(math.pi), 180)
        self.assertAlmostEqual(geometry_utils.rad_to_deg(math.pi / 2), 90)
        self.assertAlmostEqual(geometry_utils.rad_to_deg(2 * math.pi), 360)

    def test_list_to_vector3(self):
        """Test list to Vector3 conversion."""
        v_list = [1.0, 2.0, 3.0]
        vector3 = geometry_utils.list_to_vector3(v_list)
        self.assertEqual(vector3.x, 1.0)
        self.assertEqual(vector3.y, 2.0)
        self.assertEqual(vector3.z, 3.0)

    def test_list_to_vector3_assertion(self):
        """Test list_to_vector3 with invalid input."""
        with self.assertRaises(AssertionError):
            geometry_utils.list_to_vector3([1.0, 2.0])  # Only 2 elements

    def test_vector3_to_list(self):
        """Test Vector3 to list conversion."""
        vector3 = Vector3(x=1.0, y=2.0, z=3.0)
        v_list = geometry_utils.vector3_to_list(vector3)
        self.assertEqual(v_list, [1.0, 2.0, 3.0])

    def test_list_to_point(self):
        """Test list to Point conversion."""
        p_list = [1.0, 2.0, 3.0]
        point = geometry_utils.list_to_point(p_list)
        self.assertEqual(point.x, 1.0)
        self.assertEqual(point.y, 2.0)
        self.assertEqual(point.z, 3.0)

    def test_list_to_point_assertion(self):
        """Test list_to_point with invalid input."""
        with self.assertRaises(AssertionError):
            geometry_utils.list_to_point([1.0, 2.0])  # Only 2 elements

    def test_point_to_list(self):
        """Test Point to list conversion."""
        point = Point(x=1.0, y=2.0, z=3.0)
        p_list = geometry_utils.point_to_list(point)
        self.assertEqual(p_list, [1.0, 2.0, 3.0])

    def test_point_to_dict(self):
        """Test Point to dictionary conversion."""
        point = Point(x=1.0, y=2.0, z=3.0)
        p_dict = geometry_utils.point_to_dict(point)
        expected = {'x': 1.0, 'y': 2.0, 'z': 3.0}
        self.assertEqual(p_dict, expected)

    def test_list_to_quaternion(self):
        """Test list to Quaternion conversion."""
        q_list = [0.707, 0.0, 0.707, 0.0]
        quaternion = geometry_utils.list_to_quaternion(q_list)
        self.assertEqual(quaternion.w, 0.707)
        self.assertEqual(quaternion.x, 0.0)
        self.assertEqual(quaternion.y, 0.707)
        self.assertEqual(quaternion.z, 0.0)

    def test_list_to_quaternion_assertion(self):
        """Test list_to_quaternion with invalid input."""
        with self.assertRaises(AssertionError):
            geometry_utils.list_to_quaternion([1.0, 2.0, 3.0])  # Only 3 elements

    def test_quaternion_to_list(self):
        """Test Quaternion to list conversion."""
        quaternion = Quaternion(w=0.707, x=0.0, y=0.707, z=0.0)
        q_list = geometry_utils.quaternion_to_list(quaternion)
        self.assertEqual(q_list, [0.707, 0.0, 0.707, 0.0])

    def test_quaternion_to_dict(self):
        """Test Quaternion to dictionary conversion."""
        quaternion = Quaternion(w=0.707, x=0.0, y=0.707, z=0.0)
        q_dict = geometry_utils.quaternion_to_dict(quaternion)
        expected = {'qw': 0.707, 'qx': 0.0, 'qy': 0.707, 'qz': 0.0}
        self.assertEqual(q_dict, expected)

    def test_list_to_pose(self):
        """Test list to Pose conversion."""
        pose_list = [1.0, 2.0, 3.0, 0.707, 0.0, 0.707, 0.0]
        pose = geometry_utils.list_to_pose(pose_list)
        self.assertEqual(pose.position.x, 1.0)
        self.assertEqual(pose.position.y, 2.0)
        self.assertEqual(pose.position.z, 3.0)
        self.assertEqual(pose.orientation.w, 0.707)
        self.assertEqual(pose.orientation.x, 0.0)
        self.assertEqual(pose.orientation.y, 0.707)
        self.assertEqual(pose.orientation.z, 0.0)

    def test_pose_to_list(self):
        """Test Pose to list conversion."""
        pose = Pose()
        pose.position = Point(x=1.0, y=2.0, z=3.0)
        pose.orientation = Quaternion(w=0.707, x=0.0, y=0.707, z=0.0)
        pose_list = geometry_utils.pose_to_list(pose)
        expected = [1.0, 2.0, 3.0, 0.707, 0.0, 0.707, 0.0]
        self.assertEqual(pose_list, expected)

    def test_pose_to_dict(self):
        """Test Pose to dictionary conversion."""
        pose = Pose()
        pose.position = Point(x=1.0, y=2.0, z=3.0)
        pose.orientation = Quaternion(w=0.707, x=0.0, y=0.707, z=0.0)
        pose_dict = geometry_utils.pose_to_dict(pose)
        expected = {
            'x': 1.0, 'y': 2.0, 'z': 3.0,
            'qw': 0.707, 'qx': 0.0, 'qy': 0.707, 'qz': 0.0
        }
        self.assertEqual(pose_dict, expected)

    def test_quaternion_to_rpy(self):
        """Test quaternion to roll-pitch-yaw conversion."""
        # Test identity quaternion
        q = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        rpy = geometry_utils.quaternion_to_rpy(q)
        self.assertAlmostEqual(rpy.x, 0.0, places=6)
        self.assertAlmostEqual(rpy.y, 0.0, places=6)
        self.assertAlmostEqual(rpy.z, 0.0, places=6)

        # Test 90-degree rotation around Z-axis
        q = Quaternion(w=0.707, x=0.0, y=0.0, z=0.707)
        rpy = geometry_utils.quaternion_to_rpy(q)
        self.assertAlmostEqual(rpy.z, math.pi/2, places=3)

    def test_rpy_to_quaternion(self):
        """Test roll-pitch-yaw to quaternion conversion."""
        # Test zero rotation
        rpy = Vector3(x=0.0, y=0.0, z=0.0)
        q = geometry_utils.rpy_to_quaternion(rpy)
        self.assertAlmostEqual(q.w, 1.0, places=6)
        self.assertAlmostEqual(q.x, 0.0, places=6)
        self.assertAlmostEqual(q.y, 0.0, places=6)
        self.assertAlmostEqual(q.z, 0.0, places=6)

        # Test 90-degree rotation around Z-axis
        rpy = Vector3(x=0.0, y=0.0, z=math.pi/2)
        q = geometry_utils.rpy_to_quaternion(rpy)
        self.assertAlmostEqual(q.w, 0.707, places=3)
        self.assertAlmostEqual(q.x, 0.0, places=3)
        self.assertAlmostEqual(q.y, 0.0, places=3)
        self.assertAlmostEqual(q.z, 0.707, places=3)

    def test_quaternion_to_rotation_matrix(self):
        """Test quaternion to rotation matrix conversion."""
        # Test identity quaternion
        q = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        matrix = geometry_utils.quaternion_to_rotation_matrix(q)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(matrix, expected)

        # Test 90-degree rotation around Z-axis
        q = Quaternion(w=0.707, x=0.0, y=0.0, z=0.707)
        matrix = geometry_utils.quaternion_to_rotation_matrix(q)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_array_almost_equal(matrix, expected, decimal=3)

    def test_rotation_matrix_to_quaternion(self):
        """Test rotation matrix to quaternion conversion."""
        # Test identity matrix
        matrix = np.eye(3)
        q = geometry_utils.rotation_matrix_to_quaternion(matrix)
        self.assertAlmostEqual(q.w, 1.0, places=6)
        self.assertAlmostEqual(q.x, 0.0, places=6)
        self.assertAlmostEqual(q.y, 0.0, places=6)
        self.assertAlmostEqual(q.z, 0.0, places=6)

        # Test 90-degree rotation around Z-axis
        matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        q = geometry_utils.rotation_matrix_to_quaternion(matrix)
        self.assertAlmostEqual(q.w, 0.707, places=3)
        self.assertAlmostEqual(q.x, 0.0, places=3)
        self.assertAlmostEqual(q.y, 0.0, places=3)
        self.assertAlmostEqual(q.z, 0.707, places=3)

    def test_quaternion_positive(self):
        """Test quaternion positive function."""
        # Test positive quaternion (unchanged)
        q = Quaternion(w=0.707, x=0.0, y=0.707, z=0.0)
        result = geometry_utils.quaternion_positive(q)
        self.assertEqual(result.w, 0.707)
        self.assertEqual(result.x, 0.0)
        self.assertEqual(result.y, 0.707)
        self.assertEqual(result.z, 0.0)

        # Test negative quaternion (should be negated)
        q = Quaternion(w=-0.707, x=0.0, y=-0.707, z=0.0)
        result = geometry_utils.quaternion_positive(q)
        self.assertEqual(result.w, 0.707)
        self.assertEqual(result.x, 0.0)
        self.assertEqual(result.y, 0.707)
        self.assertEqual(result.z, 0.0)

    def test_quaternion_negative(self):
        """Test quaternion negative function."""
        # Test positive quaternion (should be negated)
        q = Quaternion(w=0.707, x=0.0, y=0.707, z=0.0)
        result = geometry_utils.quaternion_negative(q)
        self.assertEqual(result.w, -0.707)
        self.assertEqual(result.x, 0.0)
        self.assertEqual(result.y, -0.707)
        self.assertEqual(result.z, 0.0)

        # Test negative quaternion (unchanged)
        q = Quaternion(w=-0.707, x=0.0, y=-0.707, z=0.0)
        result = geometry_utils.quaternion_negative(q)
        self.assertEqual(result.w, -0.707)
        self.assertEqual(result.x, 0.0)
        self.assertEqual(result.y, -0.707)
        self.assertEqual(result.z, 0.0)

    def test_quaternion_negate(self):
        """Test quaternion negation."""
        q = Quaternion(w=0.707, x=0.0, y=0.707, z=0.0)
        result = geometry_utils.quaternion_negate(q)
        self.assertEqual(result.w, -0.707)
        self.assertEqual(result.x, 0.0)
        self.assertEqual(result.y, -0.707)
        self.assertEqual(result.z, 0.0)

    def test_quaternion_conjugate(self):
        """Test quaternion conjugate."""
        q = Quaternion(w=0.707, x=0.0, y=0.707, z=0.0)
        result = geometry_utils.quaternion_conjugate(q)
        self.assertEqual(result.w, 0.707)
        self.assertEqual(result.x, 0.0)
        self.assertEqual(result.y, -0.707)
        self.assertEqual(result.z, 0.0)

    def test_point_difference(self):
        """Test point difference."""
        p1 = Point(x=3.0, y=2.0, z=1.0)
        p2 = Point(x=1.0, y=1.0, z=1.0)
        result = geometry_utils.point_difference(p1, p2)
        self.assertEqual(result.x, 2.0)
        self.assertEqual(result.y, 1.0)
        self.assertEqual(result.z, 0.0)

    def test_pose_difference(self):
        """Test pose difference."""
        pose1 = Pose()
        pose1.position = Point(x=3.0, y=2.0, z=1.0)
        pose1.orientation = Quaternion(w=0.707, x=0.0, y=0.707, z=0.0)

        pose2 = Pose()
        pose2.position = Point(x=1.0, y=1.0, z=1.0)
        pose2.orientation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

        result = geometry_utils.pose_difference(pose1, pose2)

        # Check position difference
        self.assertEqual(result.position.x, 2.0)
        self.assertEqual(result.position.y, 1.0)
        self.assertEqual(result.position.z, 0.0)

        # Check orientation difference (should be the same as pose1 since pose2 is identity)
        self.assertAlmostEqual(result.orientation.w, 0.707, places=3)
        self.assertAlmostEqual(result.orientation.x, 0.0, places=3)
        self.assertAlmostEqual(result.orientation.y, 0.707, places=3)
        self.assertAlmostEqual(result.orientation.z, 0.0, places=3)

    def test_vector_to_point(self):
        """Test Vector3 to Point conversion."""
        vector = Vector3(x=1.0, y=2.0, z=3.0)
        point = geometry_utils.vector_to_point(vector)
        self.assertEqual(point.x, 1.0)
        self.assertEqual(point.y, 2.0)
        self.assertEqual(point.z, 3.0)

    def test_point_to_vector(self):
        """Test Point to Vector3 conversion."""
        point = Point(x=1.0, y=2.0, z=3.0)
        vector = geometry_utils.point_to_vector(point)
        self.assertEqual(vector.x, 1.0)
        self.assertEqual(vector.y, 2.0)
        self.assertEqual(vector.z, 3.0)

    def test_norm(self):
        """Test vector norm calculation."""
        vector = Vector3(x=3.0, y=4.0, z=0.0)
        result = geometry_utils.norm(vector)
        self.assertEqual(result, 5.0)

    def test_transform_to_matrix(self):
        """Test transform to matrix conversion."""
        transform = Transform()
        transform.translation = Vector3(x=1.0, y=2.0, z=3.0)
        transform.rotation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)  # Identity rotation

        matrix = geometry_utils.transform_to_matrix(transform)

        # Check translation
        self.assertEqual(matrix[0, 3], 1.0)
        self.assertEqual(matrix[1, 3], 2.0)
        self.assertEqual(matrix[2, 3], 3.0)

        # Check rotation (should be identity)
        rotation_matrix = matrix[:3, :3]
        np.testing.assert_array_almost_equal(rotation_matrix, np.eye(3))

    def test_matrix_to_transform(self):
        """Test matrix to transform conversion."""
        matrix = np.eye(4)
        matrix[0, 3] = 1.0
        matrix[1, 3] = 2.0
        matrix[2, 3] = 3.0

        transform = geometry_utils.matrix_to_transform(matrix)

        self.assertEqual(transform.translation.x, 1.0)
        self.assertEqual(transform.translation.y, 2.0)
        self.assertEqual(transform.translation.z, 3.0)

        # Check rotation (should be identity)
        self.assertAlmostEqual(transform.rotation.w, 1.0, places=6)
        self.assertAlmostEqual(transform.rotation.x, 0.0, places=6)
        self.assertAlmostEqual(transform.rotation.y, 0.0, places=6)
        self.assertAlmostEqual(transform.rotation.z, 0.0, places=6)

    def test_get_transformation_matrix_from_ros(self):
        """Test getting transformation matrix from ROS pose."""
        translation = Point(x=1.0, y=2.0, z=3.0)
        rotation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)  # Identity rotation

        matrix = geometry_utils.get_transformation_matrix_from_ros(translation, rotation)

        # Check translation
        self.assertEqual(matrix[0, 3], 1.0)
        self.assertEqual(matrix[1, 3], 2.0)
        self.assertEqual(matrix[2, 3], 3.0)

        # Check rotation (should be identity)
        rotation_matrix = matrix[:3, :3]
        np.testing.assert_array_almost_equal(rotation_matrix, np.eye(3))

    def test_ros_pose_to_ros_transform(self):
        """Test ROS pose to ROS transform conversion."""
        pose = Pose()
        pose.position = Point(x=1.0, y=2.0, z=3.0)
        pose.orientation = Quaternion(w=0.707, x=0.0, y=0.707, z=0.0)

        transform = geometry_utils.ros_pose_to_ros_transform(pose)

        self.assertEqual(transform.translation.x, 1.0)
        self.assertEqual(transform.translation.y, 2.0)
        self.assertEqual(transform.translation.z, 3.0)
        self.assertEqual(transform.rotation.w, 0.707)
        self.assertEqual(transform.rotation.x, 0.0)
        self.assertEqual(transform.rotation.y, 0.707)
        self.assertEqual(transform.rotation.z, 0.0)

    def test_compute_transformation_delta(self):
        """Test computation of transformation delta."""
        # Create two transformation matrices
        A = np.eye(4)
        A[0, 3] = 1.0  # Translation of 1 unit in x

        B = np.eye(4)
        B[0, 3] = 2.0  # Translation of 2 units in x

        delta = geometry_utils.compute_transformation_delta(A, B)

        # The delta should represent the transformation from A to B
        # Since A has translation (1,0,0) and B has (2,0,0), delta should have (1,0,0)
        self.assertEqual(delta[0, 3], 1.0)
        self.assertEqual(delta[1, 3], 0.0)
        self.assertEqual(delta[2, 3], 0.0)

    def test_compute_delta_translation_rotation(self):
        """Test computation of delta translation and rotation."""
        # Create a delta transformation with pure translation
        delta = np.eye(4)
        delta[0, 3] = 0.1  # 10cm translation in x

        translation, rotation = geometry_utils.compute_delta_translation_rotation(
            delta, cm_and_mrad=True)

        self.assertEqual(translation, 10.0)  # 10cm
        self.assertAlmostEqual(rotation, 0.0, places=6)  # No rotation

        # Test with rotation
        delta = np.eye(4)
        delta[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90-degree rotation

        translation, rotation = geometry_utils.compute_delta_translation_rotation(
            delta, cm_and_mrad=True)

        self.assertEqual(translation, 0.0)  # No translation
        self.assertAlmostEqual(rotation, math.pi/2 * 1000, places=3)  # 90 degrees in mrad

    def test_compute_delta_translation_rotation_meters_degrees(self):
        """Test computation of delta translation and rotation in meters and degrees."""
        # Create a delta transformation with pure translation
        delta = np.eye(4)
        delta[0, 3] = 0.1  # 10cm translation in x

        translation, rotation = geometry_utils.compute_delta_translation_rotation(
            delta, cm_and_mrad=False)

        self.assertEqual(translation, 0.1)  # 0.1 meters
        self.assertAlmostEqual(rotation, 0.0, places=6)  # No rotation

        # Test with rotation
        delta = np.eye(4)
        delta[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90-degree rotation

        translation, rotation = geometry_utils.compute_delta_translation_rotation(
            delta, cm_and_mrad=False)

        self.assertEqual(translation, 0.0)  # No translation
        self.assertAlmostEqual(rotation, 90.0, places=3)  # 90 degrees

    def test_get_pose_from_transform(self):
        """Test getting pose from transform."""
        transform = Transform()
        transform.translation = Vector3(x=1.0, y=2.0, z=3.0)
        transform.rotation = Quaternion(w=0.707, x=0.0, y=0.707, z=0.0)

        pose = geometry_utils.get_pose_from_transform(transform)

        self.assertEqual(pose.position.x, 1.0)
        self.assertEqual(pose.position.y, 2.0)
        self.assertEqual(pose.position.z, 3.0)
        self.assertEqual(pose.orientation.w, 0.707)
        self.assertEqual(pose.orientation.x, 0.0)
        self.assertEqual(pose.orientation.y, 0.707)
        self.assertEqual(pose.orientation.z, 0.0)

    def test_rotate_pose_x(self):
        """Test rotating pose around x-axis."""
        pose = Pose()
        pose.position = Point(x=1.0, y=2.0, z=3.0)
        pose.orientation = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)  # Identity rotation

        # Rotate 90 degrees around x-axis
        rotated_pose = geometry_utils.rotate_pose(pose, 90.0, 'x')

        # Position should remain the same
        self.assertEqual(rotated_pose.position.x, 1.0)
        self.assertEqual(rotated_pose.position.y, 2.0)
        self.assertEqual(rotated_pose.position.z, 3.0)

        # Orientation should be 90-degree rotation around x-axis
        # Expected quaternion for 90-degree rotation around x-axis: [0.707, 0.707, 0, 0]
        self.assertAlmostEqual(rotated_pose.orientation.w, 0.707, places=3)
        self.assertAlmostEqual(rotated_pose.orientation.x, 0.707, places=3)
        self.assertAlmostEqual(rotated_pose.orientation.y, 0.0, places=3)
        self.assertAlmostEqual(rotated_pose.orientation.z, 0.0, places=3)


if __name__ == '__main__':
    unittest.main()
