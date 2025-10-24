// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef ISAAC_MANIPULATOR_SERVERS__SEGMENT_ANYTHING_SERVER_HPP_
#define ISAAC_MANIPULATOR_SERVERS__SEGMENT_ANYTHING_SERVER_HPP_

#include <memory>
#include <string>

#include "isaac_manipulator_interfaces/action/segment_anything.hpp"
#include "isaac_ros_common/qos.hpp"
#include "isaac_ros_segment_anything2_interfaces/srv/add_objects.hpp"
#include "isaac_ros_segment_anything2_interfaces/srv/remove_object.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/point2_d.hpp"

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

class SegmentAnythingServer : public rclcpp::Node
{
public:
  explicit SegmentAnythingServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~SegmentAnythingServer();

private:
  using SegmentAnything =
    isaac_manipulator_interfaces::action::SegmentAnything;
  using AddObjects =
    isaac_ros_segment_anything2_interfaces::srv::AddObjects;
  using RemoveObject =
    isaac_ros_segment_anything2_interfaces::srv::RemoveObject;

  rclcpp_action::GoalResponse HandleGoal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const SegmentAnything::Goal> goal);
  rclcpp_action::CancelResponse HandleCancel(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<SegmentAnything>> goal_handle);
  void HandleAccepted(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<SegmentAnything>> goal_handle);
  void Execute(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<SegmentAnything>> goal_handle);

  void CallbackImg(const sensor_msgs::msg::Image::ConstSharedPtr & msg_ptr);
  void CallbackCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg_ptr);
  void CallbackSegmentationMask(const sensor_msgs::msg::Image::ConstSharedPtr & msg_ptr);
  void CallbackDetections(const vision_msgs::msg::Detection2DArray::ConstSharedPtr & msg_ptr);
  bool CallAddObjectsService(
    std_msgs::msg::Header header, bool use_point_hint,
    vision_msgs::msg::BoundingBox2D initial_hint_bbox,
    vision_msgs::msg::Point2D initial_hint_point);
  bool CallRemoveObjectService(std_msgs::msg::Header header);
  void ClearAllMsgs();

  // The user generates a geometry_msgs/Point message using RQT image view and converts it
  // into a vision_msgs/Point2D message to send to the servers. However, since SAM1 takes
  // vision_msgs/Detection2DArray as input, we use this function to convert the point to a
  // detection array.
  vision_msgs::msg::Detection2DArray PointToDetection2DArray(
    const vision_msgs::msg::Point2D & point,
    const rclcpp::Time & timestamp);

  // When using an object detection backend, a vision_msgs/BoundingBox2D message is sent to the
  // segment anything server. We use this function to convert the bounding box to a detection
  // array to produce compatible inputs for SAM1.
  vision_msgs::msg::Detection2DArray BoundingBox2DToDetection2DArray(
    const vision_msgs::msg::BoundingBox2D & bbox,
    const rclcpp::Time & timestamp);

private:
  // True if using SAM2
  bool is_sam2_;

  // Name of the action server
  std::string action_name_;
  std::string in_img_topic_name_;
  std::string out_img_topic_name_;
  std::string in_camera_info_topic_name_;
  std::string out_camera_info_topic_name_;
  std::string in_segmentation_mask_topic_name_;
  std::string in_detections_topic_name_;
  std::string out_detections_topic_name_;

  // Timeout for service calls
  int service_call_timeout_;
  int service_discovery_timeout_;

  // QOS for subscriptions and publishers
  rclcpp::QoS input_qos_;
  rclcpp::QoS result_and_output_qos_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr segmentation_mask_sub_;
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detections_sub_;

  const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub_;
  const rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_pub_;
  const rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detections_pub_;

  sensor_msgs::msg::Image::ConstSharedPtr img_msg_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr cam_info_msg_;
  sensor_msgs::msg::Image::ConstSharedPtr segmentation_mask_msg_;
  vision_msgs::msg::Detection2DArray::ConstSharedPtr detections_msg_;

  rclcpp_action::Server<SegmentAnything>::SharedPtr action_server_;
  rclcpp::Client<AddObjects>::SharedPtr add_objects_client_;
  rclcpp::Client<RemoveObject>::SharedPtr remove_object_client_;

  rclcpp::CallbackGroup::SharedPtr action_cb_group_;
  rclcpp::CallbackGroup::SharedPtr client_cb_group_;
  rclcpp::CallbackGroup::SharedPtr subscription_cb_group_;
};

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

#endif  // ISAAC_MANIPULATOR_SERVERS__SEGMENT_ANYTHING_SERVER_HPP_
