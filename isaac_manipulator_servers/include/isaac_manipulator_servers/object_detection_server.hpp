// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef ISAAC_MANIPULATOR_SERVERS__OBJECT_DETECTION_SERVER_HPP_
#define ISAAC_MANIPULATOR_SERVERS__OBJECT_DETECTION_SERVER_HPP_

#include <memory>
#include <string>

#include "isaac_manipulator_interfaces/action/detect_objects.hpp"
#include "isaac_ros_common/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

// The ObjectDetectionServer class is a ROS 2 action server that receives an image and
// detected objects in that image. It creates the frontend for an object detection
// ROS 2 graph.

class ObjectDetectionServer : public rclcpp::Node
{
public:
  explicit ObjectDetectionServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~ObjectDetectionServer() = default;

private:
  using DetectObjects = isaac_manipulator_interfaces::action::DetectObjects;

  rclcpp_action::GoalResponse HandleGoal(
    const rclcpp_action::GoalUUID & uuid, std::shared_ptr<const DetectObjects::Goal> goal);

  rclcpp_action::CancelResponse HandleCancel(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<DetectObjects>> goal_handle);

  void HandleAccepted(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<DetectObjects>> goal_handle);

  void Execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<DetectObjects>> goal_handle);

  void CallbackImg(const sensor_msgs::msg::Image::ConstSharedPtr & msg_ptr);
  void CallbackDetections(const vision_msgs::msg::Detection2DArray::ConstSharedPtr & msg_ptr);
  void CallbackCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg_ptr);

  void ClearAllMsgs();

private:
  // Name of the action server
  std::string action_name_ = "";
  std::string input_img_topic_name_ = "";
  std::string output_img_topic_name_ = "";
  std::string input_detections_topic_name_ = "";
  std::string output_detections_topic_name_ = "";
  std::string input_camera_info_topic_name_ = "";
  std::string output_camera_info_topic_name_ = "";

  // QOS for subscriptions and publishers
  rclcpp::QoS input_qos_;
  rclcpp::QoS result_and_output_qos_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detections_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

  const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub_;
  const rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;
  const rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detections_pub_;

  sensor_msgs::msg::Image::ConstSharedPtr img_msg_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_msg_;
  vision_msgs::msg::Detection2DArray::ConstSharedPtr detections_msg_;

  rclcpp::CallbackGroup::SharedPtr action_cb_group_;
  rclcpp::CallbackGroup::SharedPtr subscription_cb_group_;

  rclcpp_action::Server<DetectObjects>::SharedPtr action_server_;
};

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

#endif  // ISAAC_MANIPULATOR_SERVERS__OBJECT_DETECTION_SERVER_HPP_
