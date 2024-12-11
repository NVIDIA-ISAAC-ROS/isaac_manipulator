// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include "isaac_manipulator_servers/dope_server.hpp"


namespace nvidia
{
namespace isaac
{
namespace manipulation
{

DopeServer::DopeServer(const rclcpp::NodeOptions & options)
: Node("dope_action_server", options),
  action_name_(declare_parameter<std::string>("action_name", "dope")),
  in_img_topic_name_(declare_parameter<std::string>("in_img_topic_name", "image_color")),
  out_img_topic_name_(declare_parameter<std::string>(
      "out_img_topic_name", "dope_server/image_color")),
  in_camera_info_topic_name_(declare_parameter<std::string>(
      "in_camera_info_topic_name", "camera_info")),
  out_camera_info_topic_name_(declare_parameter<std::string>(
      "out_camera_info_topic_name", "dope_server/camera_info")),
  in_pose_estimate_topic_name_(declare_parameter<std::string>(
      "in_pose_estimate_topic_name", "poses")),
  out_pose_estimate_topic_name_(declare_parameter<std::string>(
      "out_pose_estimate_topic_name", "dope_server/poses")),
  in_detections_topic_name_(declare_parameter<std::string>(
      "in_detections_topic_name", "detections")),
  out_detections_topic_name_(declare_parameter<std::string>(
      "out_detections_topic_name", "dope_server/detections")),
  enable_2d_detections_(declare_parameter<bool>("enable_2d_detections", false)),
  sub_qos_{::isaac_ros::common::AddQosParameter(*this, "SENSOR_DATA", "sub_qos")},
  pub_qos_{::isaac_ros::common::AddQosParameter(*this, "SENSOR_DATA", "pub_qos")},
  img_pub_(
    create_publisher<sensor_msgs::msg::Image>(out_img_topic_name_, pub_qos_)),
  cam_info_pub_(
    create_publisher<sensor_msgs::msg::CameraInfo>(out_camera_info_topic_name_, pub_qos_)),
  pose_estimate_pub_(
    create_publisher<vision_msgs::msg::Detection3DArray>(out_pose_estimate_topic_name_, pub_qos_)),
  detections_pub_(
    create_publisher<vision_msgs::msg::Detection2DArray>(out_detections_topic_name_, pub_qos_)),
  img_msg_(nullptr),
  cam_info_msg_(nullptr),
  pose_estimate_msg_(nullptr),
  action_cb_group_(create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive)),
  subscription_cb_group_(create_callback_group(rclcpp::CallbackGroupType::Reentrant))
{
  action_server_ = rclcpp_action::create_server<EstimatePoseDope>(
    this,
    action_name_,
    std::bind(
      &DopeServer::HandleGoal, this, std::placeholders::_1, std::placeholders::_2),
    std::bind(&DopeServer::HandleCancel, this, std::placeholders::_1),
    std::bind(&DopeServer::HandleAccepted, this, std::placeholders::_1),
    rcl_action_server_get_default_options(),
    action_cb_group_);

  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = subscription_cb_group_;

  img_sub_ = create_subscription<sensor_msgs::msg::Image>(
    in_img_topic_name_, sub_qos_,
    std::bind(&DopeServer::CallbackImg, this, std::placeholders::_1), sub_options);

  cam_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
    in_camera_info_topic_name_, sub_qos_,
    std::bind(&DopeServer::CallbackCameraInfo, this, std::placeholders::_1), sub_options);

  pose_estimate_sub_ = create_subscription<vision_msgs::msg::Detection3DArray>(
    in_pose_estimate_topic_name_, sub_qos_,
    std::bind(&DopeServer::CallbackPoseEstimate, this, std::placeholders::_1), sub_options);

  detections_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
    in_detections_topic_name_, sub_qos_,
    std::bind(&DopeServer::CallbackDetections, this, std::placeholders::_1), sub_options);
}

rclcpp_action::GoalResponse DopeServer::HandleGoal(
  const rclcpp_action::GoalUUID & uuid, std::shared_ptr<const EstimatePoseDope::Goal> goal)
{
  RCLCPP_INFO(get_logger(), "Received goal request for DOPE pose estimation");
  (void)uuid;
  (void)goal;
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse DopeServer::HandleCancel(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<EstimatePoseDope>> goal_handle)
{
  RCLCPP_INFO(get_logger(), "Received request to cancel goal");
  (void)goal_handle;
  return rclcpp_action::CancelResponse::ACCEPT;
}

void DopeServer::HandleAccepted(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<EstimatePoseDope>> goal_handle)
{
  // this needs to return quickly to avoid blocking the executor, so spin up a new thread
  std::thread{std::bind(&DopeServer::Execute, this, std::placeholders::_1),
    goal_handle}.detach();
}

void DopeServer::Execute(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<EstimatePoseDope>> goal_handle)
{
  RCLCPP_INFO(get_logger(), "Executing goal");
  // Wait for the latest image.
  while (img_msg_ == nullptr) {
    RCLCPP_DEBUG(get_logger(), "Waiting for image");
  }
  while (cam_info_msg_ == nullptr) {
    RCLCPP_DEBUG(get_logger(), "Waiting for camera info");
  }
  // Publish the latest image
  img_pub_->publish(*img_msg_);
  // Reset the image for the next request.
  img_msg_.reset();

  // Publish the latest camera_info
  cam_info_pub_->publish(*cam_info_msg_);
  // Reset the camera_info for the next request.
  cam_info_msg_.reset();

  RCLCPP_INFO(get_logger(), "All msgs published, waiting for poses.");

  // Wait for the pose estimation result. Reset for every new request.
  while (pose_estimate_msg_ == nullptr) {
    RCLCPP_DEBUG(get_logger(), "Waiting for pose estimation result");
  }

  // Wait for the 2d detections result. Reset for every new request.
  while (detections_msg_ == nullptr && enable_2d_detections_) {
    RCLCPP_DEBUG(get_logger(), "Waiting for 2d detections result");
  }

  RCLCPP_INFO(get_logger(), "Publishing pose estimation result");

  // Todo(swani): Implement the failure detection logic here incase the pose estimation fails.
  // In that case, we should return a failure result to the client.

  auto result = std::make_shared<EstimatePoseDope::Result>();
  result->poses = *pose_estimate_msg_;
  if (enable_2d_detections_) {
    result->detections = *detections_msg_;
  }

  pose_estimate_msg_.reset();
  detections_msg_.reset();

  goal_handle->succeed(result);
}

void DopeServer::CallbackImg(const sensor_msgs::msg::Image::ConstSharedPtr & msg_ptr)
{
  img_msg_ = msg_ptr;
  RCLCPP_DEBUG(get_logger(), "[CallbackImg] Received image");
}

void DopeServer::CallbackCameraInfo(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg_ptr)
{
  cam_info_msg_ = msg_ptr;
  RCLCPP_DEBUG(get_logger(), "[CallbackCameraInfo] Received camera_info");
}

void DopeServer::CallbackPoseEstimate(
  const vision_msgs::msg::Detection3DArray::ConstSharedPtr & msg_ptr)
{
  pose_estimate_msg_ = msg_ptr;
  RCLCPP_DEBUG(get_logger(), "[CallbackPoseEstimate] Received poses");
}

void DopeServer::CallbackDetections(
  const vision_msgs::msg::Detection2DArray::ConstSharedPtr & msg_ptr)
{
  detections_msg_ = msg_ptr;
  RCLCPP_DEBUG(get_logger(), "[CallbackDetections] Received detections");
}

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac::manipulation::DopeServer)
