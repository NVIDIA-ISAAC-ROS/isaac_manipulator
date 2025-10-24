# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from geometry_msgs.msg import Pose, PoseArray, TransformStamped
import py_trees
from std_msgs.msg import Header
import tf2_ros


class ReadGraspPoses(py_trees.behaviour.Behaviour):
    """
    Read grasp poses using GraspReaderManager and store them in the blackboard.

    This behavior loads grasp poses from a YAML file for a given object class
    using GraspReaderManager, transforms them to the appropriate coordinate frame,
    and publishes them as TF frames for visualization if requested.

    Parameters
    ----------
    name : str
        Name of the behavior
    publish_grasp_poses : bool
        Whether to publish grasp poses as TF frames for visualization

    """

    def __init__(self,
                 name: str,
                 publish_grasp_poses: bool):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client()
        self.publish_grasp_poses = publish_grasp_poses
        self.tf_broadcaster = None  # Will be initialized in setup
        self.blackboard.register_key(
            key='grasp_poses', access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(
            key='grasp_reader_manager', access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key='active_obj_id', access=py_trees.common.Access.READ)
        self.blackboard.register_key(
            key='object_info_cache', access=py_trees.common.Access.READ)

    def setup(self, **kwargs):
        try:
            self.node = kwargs['node']
            # Initialize TransformBroadcaster (dynamic transforms)
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self.node)

            # Check if GraspReaderManager exists in the blackboard
            if (not self.blackboard.exists('grasp_reader_manager') or
                    self.blackboard.grasp_reader_manager is None):
                raise RuntimeError(
                    'GraspReaderManager not initialized on blackboard.')
            else:
                self.node.get_logger().info(
                    f'[{self.name}] Using existing GraspReaderManager from blackboard')

        except KeyError as e:
            error_message = f"didn't find ros2 node in setup's kwargs for {self.name}"
            raise KeyError(error_message) from e
        except Exception as e:
            error_message = f'Failed to setup grasp reader behavior: {e}'
            raise RuntimeError(error_message) from e

    def update(self):
        try:
            # Check if active_obj_id exists in object_info_cache
            if (self.blackboard.object_info_cache is None or
                    self.blackboard.active_obj_id not in self.blackboard.object_info_cache):
                self.node.get_logger().error(
                    f'[{self.name}] Object ID {self.blackboard.active_obj_id} '
                    f'not found in object_info_cache')
                return py_trees.common.Status.FAILURE

            # Get object info and check required fields
            object_info = self.blackboard.object_info_cache[self.blackboard.active_obj_id]
            object_class_id = object_info.get('class_id')
            if object_class_id is None:
                self.node.get_logger().error(
                    f"[{self.name}] class_id doesn't exist for object ID "
                    f'{self.blackboard.active_obj_id}')
                return py_trees.common.Status.FAILURE

            # Check if estimated_pose exists in cache
            estimated_pose = object_info.get('estimated_pose')
            if estimated_pose is None:
                self.node.get_logger().error(
                    f"[{self.name}] estimated_pose doesn't exist for object ID "
                    f'{self.blackboard.active_obj_id}. Pose estimation must be completed first.')
                return py_trees.common.Status.FAILURE

            # Extract pose dict, convert to 4x4 matrix
            position = estimated_pose.get('position')
            orientation = estimated_pose.get('orientation')
            if position is None or orientation is None:
                self.node.get_logger().error(
                    f"[{self.name}] estimated_pose missing 'position' or 'orientation'")
                return py_trees.common.Status.FAILURE

            # orientation expected as [x,y,z,w]
            quat_dict = {
                'x': orientation[0],
                'y': orientation[1],
                'z': orientation[2],
                'w': orientation[3]
            }

            grasp_manager = self.blackboard.grasp_reader_manager.get_grasp_reader(object_class_id)
            world_pose_object = grasp_manager.get_transformation_matrix(quat_dict, position)
            grasp_poses = grasp_manager.get_pose_for_pick_task_from_cached_pose(world_pose_object)

            self.node.get_logger().info(
                f'[{self.name}] Successfully loaded {len(grasp_poses)} grasp poses for '
                f'object_id={self.blackboard.active_obj_id}, class_id={object_class_id}')

            poses_arr = PoseArray()
            poses_arr.header.frame_id = 'base_link'

            for i, grasp_pose in enumerate(grasp_poses):
                pose = Pose()
                pose.position.x = grasp_pose.position.x
                pose.position.y = grasp_pose.position.y
                pose.position.z = grasp_pose.position.z
                pose.orientation.w = grasp_pose.quaternion.w
                pose.orientation.x = grasp_pose.quaternion.x
                pose.orientation.y = grasp_pose.quaternion.y
                pose.orientation.z = grasp_pose.quaternion.z
                poses_arr.poses.append(pose)
                if self.publish_grasp_poses:
                    frame_name = f'grasp_frame_{i}'
                    self._publish_grasp_pose(pose, frame_name)

            if self.publish_grasp_poses:
                self.node.get_logger().info(
                    f'[{self.name}] Published {len(grasp_poses)} grasp frame transforms for '
                    f'object_id={self.blackboard.active_obj_id}')

            self.blackboard.grasp_poses = poses_arr

        except Exception as e:
            self.node.get_logger().error(
                f'[{self.name}] Failed to read grasp poses for '
                f'object_id={self.blackboard.active_obj_id}: {e}')
            return py_trees.common.Status.FAILURE

        return py_trees.common.Status.SUCCESS

    def _publish_grasp_pose(self, pose: Pose, frame_name: str):
        """Publish a single grasp pose as a dynamic transform."""
        transform_stamped = TransformStamped()
        transform_stamped.header = Header()
        transform_stamped.header.stamp = self.node.get_clock().now().to_msg()
        transform_stamped.header.frame_id = 'base_link'
        transform_stamped.child_frame_id = frame_name

        # Assign the position and orientation from the grasp pose
        transform_stamped.transform.translation.x = pose.position.x
        transform_stamped.transform.translation.y = pose.position.y
        transform_stamped.transform.translation.z = pose.position.z
        transform_stamped.transform.rotation.x = pose.orientation.x
        transform_stamped.transform.rotation.y = pose.orientation.y
        transform_stamped.transform.rotation.z = pose.orientation.z
        transform_stamped.transform.rotation.w = pose.orientation.w

        # Publish the transform
        self.tf_broadcaster.sendTransform(transform_stamped)
