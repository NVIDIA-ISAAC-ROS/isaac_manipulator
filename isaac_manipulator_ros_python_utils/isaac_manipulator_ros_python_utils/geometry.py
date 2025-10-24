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
from typing import Dict, List, Tuple

import cv2
from geometry_msgs.msg import Point, Pose, Quaternion, Transform, Vector3
import numpy as np
from scipy.spatial.transform import Rotation as R


def s_to_ns(s: float) -> int:
    return s * 1e9


def ns_to_s(ns: int) -> float:
    return ns / 1e9


def deg_to_rad(d: float) -> float:
    return d * math.pi / 180.0


def rad_to_deg(r: float) -> float:
    return r * 180.0 / math.pi


def list_to_vector3(v: List[float]) -> Vector3:
    assert len(v) == 3
    return Vector3(x=v[0], y=v[1], z=v[2])


def vector3_to_list(v: Vector3) -> List[float]:
    return [v.x, v.y, v.z]


def list_to_point(p: List[float]) -> Point:
    assert len(p) == 3
    return Point(x=p[0], y=p[1], z=p[2])


def point_to_list(p: Point) -> List[float]:
    return [p.x, p.y, p.z]


def point_to_dict(p: Point) -> Dict:
    return {'x': p.x, 'y': p.y, 'z': p.z}


def list_to_quaternion(q: List[float]) -> Quaternion:
    assert len(q) == 4
    return Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])


def quaternion_to_list(q: Quaternion) -> List[float]:
    return [q.w, q.x, q.y, q.z]


def quaternion_to_dict(q: Quaternion) -> Dict:
    return {'qw': q.w, 'qx': q.x, 'qy': q.y, 'qz': q.z}


def list_to_pose(pose: List[float]) -> Pose:
    return Pose(position=list_to_point(pose[:3]), orientation=list_to_quaternion(pose[3:]))


def pose_to_list(pose: Pose) -> List[float]:
    return point_to_list(pose.position) + quaternion_to_list(pose.orientation)


def pose_to_dict(pose: Pose) -> Dict:
    return point_to_dict(pose.position) | quaternion_to_dict(pose.orientation)


def quaternion_to_rpy(q: Quaternion) -> Vector3:
    # roll (x-axis rotation)
    sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
    x = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = math.sqrt(1 + 2 * (q.w * q.y - q.x * q.z))
    cosp = math.sqrt(1 - 2 * (q.w * q.y - q.x * q.z))
    y = 2 * math.atan2(sinp, cosp) - math.pi / 2

    # yaw (z-axis rotation)
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    z = math.atan2(siny_cosp, cosy_cosp)

    return Vector3(x=x, y=y, z=z)


def rpy_to_quaternion(v: Vector3) -> Quaternion:
    cr = math.cos(v.x * 0.5)
    sr = math.sin(v.x * 0.5)
    cp = math.cos(v.y * 0.5)
    sp = math.sin(v.y * 0.5)
    cy = math.cos(v.z * 0.5)
    sy = math.sin(v.z * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return Quaternion(w=w, x=x, y=y, z=z)


def quaternion_to_rotation_matrix(quaternion: Quaternion) -> np.ndarray:
    q = np.array([quaternion.w, quaternion.x, quaternion.y, quaternion.z])
    n = np.dot(q, q)
    if n < 1e-7:
        return np.eye(3)

    q /= np.sqrt(n)
    qw, qx, qy, qz = q

    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]
    ])


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> Quaternion:
    m = matrix
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return Quaternion(w=w, x=x, y=y, z=z)


def quaternion_positive(q: Quaternion) -> Quaternion:
    return quaternion_negate(q) if q.w < 0 else q


def quaternion_negative(q: Quaternion) -> Quaternion:
    return q if q.w < 0 else quaternion_negate(q)


def quaternion_negate(q: Quaternion) -> Quaternion:
    return Quaternion(w=-q.w, x=-q.x, y=-q.y, z=-q.z)


def quaternion_conjugate(q: Quaternion) -> Quaternion:
    return Quaternion(w=q.w, x=-q.x, y=-q.y, z=-q.z)


def quaternion_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
    w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y
    y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x
    z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w

    return Quaternion(w=w, x=x, y=y, z=z)


def quaternion_difference(q1: Quaternion, q2: Quaternion) -> Quaternion:
    return quaternion_multiply(q1, quaternion_conjugate(q2))


def point_difference(p1: Point, p2: Point) -> Point:
    x = p1.x - p2.x
    y = p1.y - p2.y
    z = p1.z - p2.z

    return Point(x=x, y=y, z=z)


def pose_difference(p1: Pose, p2: Pose) -> Pose:
    position = point_difference(p1.position, p2.position)
    orientation = quaternion_difference(p1.orientation, p2.orientation)

    return Pose(position=position, orientation=orientation)


def vector_to_point(v: Vector3) -> Point:
    return Point(x=v.x, y=v.y, z=v.z)


def point_to_vector(p: Point) -> Vector3:
    return Vector3(x=p.x, y=p.y, z=p.z)


def norm(v: Vector3) -> float:
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def transform_to_matrix(transform: Transform) -> np.ndarray:
    """
    Convert a ROS transform to a transformation matrix.

    Args
    ----
        transform (Transform): The ROS transform.

    Returns
    -------
        numpy.ndarray: The transformation matrix (4x4).

    """
    rot = quaternion_to_rotation_matrix(transform.rotation)
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [transform.translation.x, transform.translation.y, transform.translation.z]

    return T


def matrix_to_transform(matrix: np.ndarray) -> Transform:
    translation = Vector3(x=matrix[0, 3], y=matrix[1, 3], z=matrix[2, 3])
    rotation = rotation_matrix_to_quaternion(matrix[:3, :3])

    return Transform(translation=translation, rotation=rotation)


def get_transformation_matrix_from_ros(translation_ros: Point | Vector3,
                                       rotation_ros: Quaternion) -> np.ndarray:
    """
    Get the transformation matrix from ROS pose.

    Args
    ----
        translation_ros (Point | Vector3): The translation of the pose.
        rotation_ros (Quaternion): The rotation of the pose.

    Returns
    -------
        numpy.ndarray: The transformation matrix (4x4).

    """
    T = np.eye(4, 4)
    T[:3, 3] = np.asarray(
        (translation_ros.x, translation_ros.y, translation_ros.z))
    T[:3, :3] = R.from_quat(
        (rotation_ros.x, rotation_ros.y, rotation_ros.z, rotation_ros.w)).as_matrix()
    return T.astype(np.float64)


def ros_pose_to_ros_transform(pose: Pose) -> Transform:
    """
    Convert a ROS pose to a ROS transform.

    Args
    ----
        pose (Pose): The ROS pose.

    Returns
    -------
        Transform: The ROS transform.

    """
    transform = Transform()
    transform.translation.x = pose.position.x
    transform.translation.y = pose.position.y
    transform.translation.z = pose.position.z
    transform.rotation.x = pose.orientation.x
    transform.rotation.y = pose.orientation.y
    transform.rotation.z = pose.orientation.z
    transform.rotation.w = pose.orientation.w

    return transform


def pose_to_matrix(pose: Pose) -> np.ndarray:
    """
    Convert a ROS Pose to a 4x4 homogeneous transformation matrix.

    Args
    ----
        pose (Pose): ROS Pose message

    Returns
    -------
        numpy.ndarray: 4x4 transformation matrix

    """
    return transform_to_matrix(ros_pose_to_ros_transform(pose))


def matrix_to_ros_translation_quaternion(matrix: np.ndarray) -> tuple[Vector3, Quaternion]:
    """
    Convert a 4x4 homogeneous transformation matrix to ROS Vector3 and Quaternion.

    Args
    ----
        matrix (numpy.ndarray): 4x4 transformation matrix

    Returns
    -------
        tuple[Vector3, Quaternion]: translation and rotation

    """
    t = matrix_to_transform(matrix)
    return t.translation, t.rotation


def compute_transformation_delta(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute the delta transformation between two transformation matrices.

    Args
    ----
        A (numpy.ndarray): The first transformation matrix (4x4).
        B (numpy.ndarray): The second transformation matrix (4x4).

    Returns
    -------
        numpy.ndarray: The delta transformation matrix (4x4).

    """
    # Compute the inverse of A
    A_inv = np.linalg.inv(A)

    # Compute the delta transformation
    delta = np.dot(A_inv, B)

    return delta


def compute_delta_translation_rotation(delta: np.ndarray, cm_and_mrad: bool = True
                                       ) -> Tuple[float, float]:
    """
    Compute the translation and rotation (in degrees) from a delta transformation matrix.

    Args
    ----
        delta (numpy.ndarray): The delta transformation matrix (4x4).
        cm_and_mrad (bool): Whether to return the translation and rotation in centimeters
           and milliradians.

    Returns
    -------
        Tuple[float, float]: The translation and rotation in meters and degrees.

    """
    # Extract the translation vector (in meters)
    translation = delta[:3, 3]

    # Extract the rotation matrix
    rotation_matrix = delta[:3, :3]

    # OpenCV's Rodrigues function returns rotation vector, which is axis-angle representation
    rvec, _ = cv2.Rodrigues(rotation_matrix)

    # Convert radians to degrees
    angle = np.linalg.norm(rvec.flatten())
    angles_deg = np.degrees(angle)

    if cm_and_mrad:
        # Convert rotation from radians to milliradians
        angle_mrad = angle * 1000  # 1 radian = 1000 milliradians
        return np.linalg.norm(translation) * 100, angle_mrad
    else:
        return np.linalg.norm(translation), angles_deg


def get_pose_from_transform(transform: Transform) -> Pose:
    """
    Get a pose from a transformation matrix.

    Args
    ----
        transform (Transform): The transformation matrix.

    Returns
    -------
        Pose: The pose.

    """
    pose = Pose()
    pose.position.x = transform.translation.x
    pose.position.y = transform.translation.y
    pose.position.z = transform.translation.z
    pose.orientation.x = transform.rotation.x
    pose.orientation.y = transform.rotation.y
    pose.orientation.z = transform.rotation.z
    pose.orientation.w = transform.rotation.w

    return pose


def rotate_pose(pose: Pose, angle: float, axis: str) -> Pose:
    """
    Rotate a pose around the x-axis by a given angle.

    Args
    ----
        pose (Pose): Input pose to rotate
        angle (float): Rotation angle in degrees
        axis (str): Axis to rotate around (x, y, z)

    Returns
    -------
        Pose: New pose with the rotation applied

    """
    # Create rotation around x-axis using the provided angle
    rotation_x = R.from_euler(axis, angle, degrees=True)

    # Get current rotation from pose
    current_rotation = R.from_quat([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    ])

    # Apply the x-axis rotation to the current rotation
    new_rotation = rotation_x * current_rotation

    # Get the new quaternion (scipy returns [x, y, z, w] format)
    new_quat = new_rotation.as_quat()

    # Create new pose with rotated orientation
    new_pose = Pose()
    new_pose.position = pose.position  # Position stays the same
    new_pose.orientation = Quaternion(
        x=new_quat[0],  # x component
        y=new_quat[1],  # y component
        z=new_quat[2],  # z component
        w=new_quat[3]   # w component
    )

    return new_pose
