#!/usr/bin/env python3
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from dataclasses import dataclass
import os
from typing import Dict, List, Tuple

from geometry_msgs.msg import Quaternion, TransformStamped, Vector3
from isaac_manipulator_ros_python_utils.config import SupportedObjectsConfig
import numpy as np
import rclpy
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer

import yaml


@dataclass
class Transformation:
    quaternion: Dict
    position: List


def lookup_transform_from_tf(
        parent_frame: str, object_frame_name: str, tf_buffer: Buffer) -> TransformStamped:
    try:
        object_transform_stamped = tf_buffer.lookup_transform(
            parent_frame, object_frame_name, rclpy.time.Time())
    except Exception as ex:
        raise ValueError(f'Could not transform {parent_frame} to {object_frame_name}: {ex}')
    return object_transform_stamped


class GraspReader:

    def __init__(self, grasp_file_path: str):
        self.grasp_file_path = grasp_file_path
        self._read_grasp_data()

    def _read_grasp_data(self):
        """Read grasp data from a YAML file."""
        try:
            with open(self.grasp_file_path, 'r') as file:
                self.grasp_data = yaml.safe_load(file)
        except Exception as e:
            raise Exception(f'Error reading YAML file: {e}')

    def _get_object_pose_gripper(self) -> List[Transformation]:
        """
        Get the pose of the gripper w.r.t the object.

        Raises
        ------
            ValueError: Error if grasp file is empty.

        Returns
        -------
            List[Transformation]: List of transformations stored in the grasp file.

        """
        if not self.grasp_data:
            raise ValueError('Grasp file is empty !')
        transformations = []
        # Extract data for the first grasp
        for grasp_key, grasp in self.grasp_data['grasps'].items():
            position = grasp['position']
            orientation = grasp['orientation']

            # Convert quaternion to euler (roll, pitch, yaw) if necessary
            quat = {
                'x': orientation['xyz'][0],
                'y': orientation['xyz'][1],
                'z': orientation['xyz'][2],
                'w': orientation['w'],
            }
            transformations.append(Transformation(quaternion=quat, position=position))
        return transformations

    def _compute_grasp_poses_from_object_pose(
            self, world_pose_object: np.ndarray) -> List[Transformation]:
        """
        Compute grasp poses in world frame given object pose in world frame.

        This is the common logic shared between get_pose_for_pick_task and
        get_pose_for_pick_task_from_cached_pose.

        Args
        ----
            world_pose_object (np.ndarray): 4x4 transformation matrix representing
                object pose in world frame

        Returns
        -------
            List[Transformation]: List of gripper poses in world frame for all grasp poses

        """
        transformations = []
        # Get all pre-recorded grasp poses (object poses relative to gripper frame)
        object_pose_gripper_list = self._get_object_pose_gripper()
        for object_pose_gripper_transform in object_pose_gripper_list:
            object_pose_gripper = self.get_transformation_matrix_from_transformation(
                object_pose_gripper_transform)

            # Compute the world pose of the tool frame
            world_pose_gripper = world_pose_object @ object_pose_gripper

            # Convert the matrix back to a Pose message and return
            translation_ros, quaternion_ros = self.matrix_to_ros_translation_quaternion(
                world_pose_gripper)
            transformations.append(Transformation(
                position=translation_ros, quaternion=quaternion_ros))
        return transformations

    def get_transformation_matrix_from_ros_(self, transform: TransformStamped) -> np.ndarray:
        """
        Get transformation matrix from ROS type.

        Args
        ----
            transform (TransformStamped): ROS transform from TF

        Returns
        -------
            np.ndarray: Transformation matrix in numpy

        """
        translation_ros, rotation_ros = transform.translation, transform.rotation
        T = np.eye(4, 4)
        T[:3, 3] = np.asarray((translation_ros.x, translation_ros.y, translation_ros.z))
        T[:3, :3] = R.from_quat((rotation_ros.x, rotation_ros.y,
                                rotation_ros.z, rotation_ros.w)).as_matrix()
        return T.astype(np.float64)

    def get_transformation_matrix(self, quaternion_dict, translation) -> np.ndarray:
        """
        Inverts the transformation defined by a quaternion (in dictionary format) and translation.

        Args
        ----
            quaternion_dict (dict): The original quaternion with keys 'x', 'y', 'z', 'w'.
            translation (list or np.array): The original translation [tx, ty, tz].

        Returns
        -------
            tuple: A tuple containing the inverted quaternion (in dictionary format) and
                translation

        """
        # Convert the dictionary quaternion to a list format
        quaternion = [
            quaternion_dict['x'],
            quaternion_dict['y'],
            quaternion_dict['z'],
            quaternion_dict['w']
        ]

        # Convert Quaternion to Rotation Matrix
        rotation = R.from_quat(quaternion)
        rotation_matrix = rotation.as_matrix()

        # Combine Rotation Matrix and Translation into a Transformation Matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation
        return transformation_matrix

    def get_transformation_matrix_from_transformation(
            self, transformation: Transformation) -> np.ndarray:
        """
        Convert a Transformation object into a 4x4 numpy transformation matrix.

        Args
        ----
            transformation (Transformation): Contains rotation and translation

        Returns
        -------
            np.ndarray: 4x4 transformation matrix

        """
        quaternion = transformation.quaternion
        position = transformation.position
        return self.get_transformation_matrix(quaternion, position)

    def matrix_to_ros_translation_quaternion(self, matrix: np.ndarray
                                             ) -> Tuple[Vector3, Quaternion]:
        """
        Transform the numpy 4x4 transformation matrix to ROS friendly objects.

        Args
        ----
            matrix (np.ndarray): Numpy array

        Returns
        -------
            Tuple[Vector3, Quaternion]: A vector and a quaternion.

        """
        # Extract translation
        translation_vector = matrix[:3, 3]
        translation_ros = Vector3()
        translation_ros.x = translation_vector[0]
        translation_ros.y = translation_vector[1]
        translation_ros.z = translation_vector[2]

        # Extract rotation matrix and convert to quaternion
        rotation_matrix = matrix[:3, :3]
        rotation_scipy = R.from_matrix(rotation_matrix)
        quaternion_scipy = rotation_scipy.as_quat()  # Returns [x, y, z, w]

        quaternion_ros = Quaternion()
        quaternion_ros.x = quaternion_scipy[0]
        quaternion_ros.y = quaternion_scipy[1]
        quaternion_ros.z = quaternion_scipy[2]
        quaternion_ros.w = quaternion_scipy[3]

        return translation_ros, quaternion_ros

    def get_grasp_pose_object(self, index: int,
                              gripper_frame: str = 'gripper_frame',
                              grasp_frame: str = 'grasp_frame',
                              tf_buffer: Buffer = None) -> Transformation:
        """
        Get the pose of the object with respect to grasp frame.

        Args
        ----
            index (int): index of grasp poses in file that was selected.
            gripper_frame (str): The frame w.r.t to which the poses were recorded unsing Grasp
                editor
            grasp_frame (str): The grasp frame that is at an offset w.r.t to the gripper frame such
                that the TF aligns between the gripper finger tips
            tf_buffer (Buffer): TF buffer that stores list of transforms

        Returns
        -------
            Transformation: Transformation object representing `grasp_pose_object`.

        """
        # Get all pre-recorded grasp poses (object poses relative to gripper frame)
        object_pose_gripper_list = self._get_object_pose_gripper()
        if index > len(object_pose_gripper_list):
            raise KeyError(
                f'{index} is out of bound of arr of length {len(object_pose_gripper_list)}')
        selected_object_pose_gripper = object_pose_gripper_list[index]
        object_pose_gripper = self.get_transformation_matrix_from_transformation(
            selected_object_pose_gripper)

        gripper_pose_grasp_transform_stamped = lookup_transform_from_tf(
            gripper_frame, grasp_frame, tf_buffer)

        # Convert TransformStamped to a 4x4 matrix
        gripper_pose_grasp = self.get_transformation_matrix_from_ros_(
            gripper_pose_grasp_transform_stamped.transform)

        # Get grasp_pose_object
        object_pose_grasp = object_pose_gripper @ gripper_pose_grasp

        grasp_pose_object = np.linalg.inv(object_pose_grasp)
        translation_ros, quaternion_ros = self.matrix_to_ros_translation_quaternion(
            grasp_pose_object)

        return Transformation(quaternion=quaternion_ros, position=translation_ros)

    def get_pose_for_pick_task(self, world_frame: str = 'base_link',
                               object_frame_name: str = 'detected_object1',
                               tf_buffer: Buffer = None) -> List[Transformation]:
        """
        Get the pose of gripper w.r.t world frame ( `world_pose_gripper`.).

        Args
        ----
            world_frame (str, optional): World frame. Defaults to 'base_link'.
            object_frame_name (str, optional): Object frame name. Defaults to 'detected_object1'.
            tf_buffer (Buffer, optional): T buffer. Defaults to None.

        Raises
        ------
            ValueError: Raises error if TF is not available

        Returns
        -------
            List[Transformation]: List of poses in the grasp file

        """
        # Parse grasp to get the object pose with respect to the gripper
        if tf_buffer is None:
            raise ValueError('TF Buffer cannot be None')

        # Get the transform from the parent frame to the object frame
        object_transform_stamped = lookup_transform_from_tf(
            world_frame, object_frame_name, tf_buffer)

        # Convert TransformStamped to a 4x4 matrix
        world_pose_object = self.get_transformation_matrix_from_ros_(
            object_transform_stamped.transform)

        # Use common logic to compute grasp poses
        return self._compute_grasp_poses_from_object_pose(world_pose_object)

    def get_pose_for_pick_task_from_cached_pose(self, world_pose_object: np.ndarray
                                                ) -> List[Transformation]:
        """
        Get the pose of gripper w.r.t world frame using provided object pose matrix.

        Args
        ----
            world_pose_object (np.ndarray): 4x4 transformation matrix representing
                object pose in world frame.

        Returns
        -------
            List[Transformation]: List of gripper poses in world frame for all grasp poses

        """
        # Validate matrix shape
        if not isinstance(world_pose_object, np.ndarray) or world_pose_object.shape != (4, 4):
            raise ValueError('world_pose_object must be a 4x4 numpy array')

        # Use common logic to compute grasp poses
        return self._compute_grasp_poses_from_object_pose(world_pose_object)


class GraspReaderManager:
    """Manager for handling GraspReader instances for different object classes."""

    def __init__(self, supported_objects_config: SupportedObjectsConfig):
        """
        Initialize GraspReaderManager with object metadata configuration.

        Args:
        ----
        supported_objects_config: SupportedObjectsConfig containing supported objects

        """
        self.supported_objects_config = supported_objects_config
        self._grasp_readers: Dict[str, GraspReader] = {}

    def get_grasp_reader(self, class_id: str) -> GraspReader:
        """
        Get or create a GraspReader for the specified object class ID.

        Args
        ----
            class_id (str): Object class ID

        Raises
        ------
            ValueError: Raises error if no configuration found for object class ID
                or no grasp file path specified for object class ID
            FileNotFoundError: Raises error if grasp file path does not exist

        Returns
        -------
            GraspReader: GraspReader instance for the specified object class ID

        """
        if class_id not in self._grasp_readers:
            supported_object = self.supported_objects_config.get_object_by_class_id(class_id)
            if supported_object is None:
                raise ValueError(f'No configuration found for object class ID: {class_id}')

            if not supported_object.grasp_file_path:
                raise ValueError(f'No grasp file path specified for object class ID: {class_id}')

            if not os.path.exists(supported_object.grasp_file_path):
                raise FileNotFoundError(
                    f'Grasp file not found at path: {supported_object.grasp_file_path}')

            # Initialize GraspReader for this class
            self._grasp_readers[class_id] = GraspReader(supported_object.grasp_file_path)

        return self._grasp_readers[class_id]
