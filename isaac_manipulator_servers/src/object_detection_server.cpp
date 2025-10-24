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


#include "isaac_manipulator_servers/object_detection_server.hpp"
#include "geometry_msgs/msg/point.hpp"


namespace nvidia
{
namespace isaac
{
namespace manipulation
{

ObjectDetectionServer::ObjectDetectionServer(const rclcpp::NodeOptions & options)
: Node("object_detection_action_server", options),
  action_name_(declare_parameter<std::string>("action_name", "object_detection")),
  input_img_topic_name_(declare_parameter<std::string>("input_img_topic_name", "image_color")),
  output_img_topic_name_(declare_parameter<std::string>(
      "output_img_topic_name", "object_detection_server/image_color")),
  input_detections_topic_name_(declare_parameter<std::string>(
      "input_detections_topic_name", "detections_output")),
  output_detections_topic_name_(declare_parameter<std::string>(
      "output_detections_topic_name", "object_detection_server/detections_output")),
  input_camera_info_topic_name_(declare_parameter<std::string>(
      "input_camera_info_topic_name", "camera_info")),
  output_camera_info_topic_name_(declare_parameter<std::string>(
      "output_camera_info_topic_name", "object_detection_server/camera_info")),
  input_qos_{::isaac_ros::common::AddQosParameter(
      *this, "SENSOR_DATA", "input_qos")},
  result_and_output_qos_{::isaac_ros::common::AddQosParameter(
      *this, "DEFAULT", "result_and_output_qos")},
  img_pub_(
    create_publisher<sensor_msgs::msg::Image>(output_img_topic_name_, result_and_output_qos_)),
  camera_info_pub_(
    create_publisher<sensor_msgs::msg::CameraInfo>(
      output_camera_info_topic_name_, result_and_output_qos_)),
  detections_pub_(
    create_publisher<vision_msgs::msg::Detection2DArray>(
      output_detections_topic_name_, result_and_output_qos_)),
  img_msg_(nullptr),
  detections_msg_(nullptr),
  action_cb_group_(create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive)),
  subscription_cb_group_(create_callback_group(rclcpp::CallbackGroupType::Reentrant))
{
  action_server_ = rclcpp_action::create_server<DetectObjects>(
    this,
    action_name_,
    std::bind(
      &ObjectDetectionServer::HandleGoal, this, std::placeholders::_1, std::placeholders::_2),
    std::bind(&ObjectDetectionServer::HandleCancel, this, std::placeholders::_1),
    std::bind(&ObjectDetectionServer::HandleAccepted, this, std::placeholders::_1),
    rcl_action_server_get_default_options(),
    action_cb_group_);

  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = subscription_cb_group_;

  img_sub_ = create_subscription<sensor_msgs::msg::Image>(
    input_img_topic_name_, input_qos_,
    std::bind(&ObjectDetectionServer::CallbackImg, this, std::placeholders::_1), sub_options);
  camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
    input_camera_info_topic_name_, input_qos_,
    std::bind(&ObjectDetectionServer::CallbackCameraInfo, this, std::placeholders::_1),
    sub_options);
  detections_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
    input_detections_topic_name_, result_and_output_qos_,
    std::bind(&ObjectDetectionServer::CallbackDetections, this, std::placeholders::_1),
    sub_options);
}

rclcpp_action::GoalResponse ObjectDetectionServer::HandleGoal(
  const rclcpp_action::GoalUUID & uuid, std::shared_ptr<const DetectObjects::Goal> goal)
{
  RCLCPP_INFO(get_logger(), "Received goal request for object detection");
  (void)uuid;
  (void)goal;
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse ObjectDetectionServer::HandleCancel(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<DetectObjects>> goal_handle)
{
  RCLCPP_INFO(get_logger(), "Received request to cancel goal");
  (void)goal_handle;
  ClearAllMsgs();
  return rclcpp_action::CancelResponse::ACCEPT;
}

void ObjectDetectionServer::HandleAccepted(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<DetectObjects>> goal_handle)
{
  // this needs to return quickly to avoid blocking the executor, so spin up a new thread
  std::thread{std::bind(&ObjectDetectionServer::Execute, this, std::placeholders::_1),
    goal_handle}.detach();
}

void ObjectDetectionServer::Execute(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<DetectObjects>> goal_handle)
{
  auto result = std::make_shared<DetectObjects::Result>();

  RCLCPP_INFO(get_logger(), "Executing goal object detection");

  // Wait for the latest image and camera info
  while ((img_msg_ == nullptr || camera_info_msg_ == nullptr) &&
    goal_handle->is_executing() && !goal_handle->is_canceling())
  {
    RCLCPP_DEBUG_ONCE(get_logger(), "Waiting for image and camera info");
  }

  if (goal_handle->is_canceling()) {
    goal_handle->canceled(result);
    RCLCPP_INFO(this->get_logger(), "Goal Canceled for action DetectObjects");
    return;
  }

  if (img_msg_ != nullptr) {
    // Publish the latest image to the implementation of object detection
    img_pub_->publish(*img_msg_);
    // Reset the image for the next request.
    img_msg_.reset();

    RCLCPP_INFO(get_logger(), "Image Published waiting for detections");
  }

  if (camera_info_msg_ != nullptr) {
    camera_info_pub_->publish(*camera_info_msg_);
    camera_info_msg_.reset();
    RCLCPP_INFO(get_logger(), "Camera Info Published");
  } else {
    RCLCPP_ERROR(get_logger(), "Camera Info is null");
  }

  // Wait for the object detection result. Reset for every new request.
  while (detections_msg_ == nullptr && goal_handle->is_executing() &&
    (!goal_handle->is_canceling()))
  {
    RCLCPP_DEBUG_ONCE(get_logger(), "Waiting for detections");
  }

  if (goal_handle->is_canceling()) {
    goal_handle->canceled(result);
    RCLCPP_INFO(this->get_logger(), "Goal Canceled for action DetectObjects");
    return;
  }

  if (detections_msg_ != nullptr) {
    RCLCPP_INFO(get_logger(), "Publishing detections");
    result->detections = *detections_msg_;
    detections_msg_.reset();

    goal_handle->succeed(result);
  }
}

void ObjectDetectionServer::CallbackImg(const sensor_msgs::msg::Image::ConstSharedPtr & msg_ptr)
{
  img_msg_ = msg_ptr;
  RCLCPP_DEBUG_ONCE(get_logger(), "[CallbackImg] Received image");
}

void ObjectDetectionServer::CallbackCameraInfo(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg_ptr)
{
  camera_info_msg_ = msg_ptr;
  RCLCPP_DEBUG_ONCE(get_logger(), "[CallbackCameraInfo] Received Camera Info");
}

void ObjectDetectionServer::CallbackDetections(
  const vision_msgs::msg::Detection2DArray::ConstSharedPtr & msg_ptr)
{
  detections_msg_ = msg_ptr;
  RCLCPP_DEBUG_ONCE(get_logger(), "[CallbackDetections] Received Detections");
}

void ObjectDetectionServer::ClearAllMsgs()
{
  img_msg_.reset();
  detections_msg_.reset();
}

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac::manipulation::ObjectDetectionServer)
