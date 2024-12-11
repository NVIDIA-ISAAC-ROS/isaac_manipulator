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


#include "isaac_manipulator_servers/foundation_pose_server.hpp"


namespace nvidia
{
namespace isaac
{
namespace manipulation
{

FoundationPoseServer::FoundationPoseServer(const rclcpp::NodeOptions & options)
: Node("foundation_pose_action_server", options),
  action_name_(declare_parameter<std::string>("action_name", "estimate_pose_foundation_pose")),
  in_img_topic_name_(declare_parameter<std::string>("in_img_topic_name", "image_color")),
  out_img_topic_name_(declare_parameter<std::string>(
      "out_img_topic_name", "foundation_pose_server/image_color")),
  in_camera_info_topic_name_(declare_parameter<std::string>(
      "in_camera_info_topic_name", "camera_info")),
  out_camera_info_topic_name_(declare_parameter<std::string>(
      "out_camera_info_topic_name", "foundation_pose_server/camera_info")),
  in_depth_topic_name_(declare_parameter<std::string>("in_depth_topic_name", "depth")),
  out_depth_topic_name_(declare_parameter<std::string>(
      "out_depth_topic_name", "foundation_pose_server/depth")),
  out_bbox_topic_name_(declare_parameter<std::string>(
      "out_bbox_topic_name", "foundation_pose_server/bbox")),
  in_pose_estimate_topic_name_(declare_parameter<std::string>(
      "in_pose_estimate_topic_name", "poses")),
  out_pose_estimate_topic_name_(declare_parameter<std::string>(
      "out_pose_estimate_topic_name", "foundation_pose_server/poses")),
  sub_qos_{::isaac_ros::common::AddQosParameter(*this, "SENSOR_DATA", "sub_qos")},
  pub_qos_{::isaac_ros::common::AddQosParameter(*this, "SENSOR_DATA", "pub_qos")},
  img_pub_(create_publisher<sensor_msgs::msg::Image>(out_img_topic_name_, pub_qos_)),
  cam_info_pub_(
    create_publisher<sensor_msgs::msg::CameraInfo>(out_camera_info_topic_name_, pub_qos_)),
  depth_pub_(create_publisher<sensor_msgs::msg::Image>(out_depth_topic_name_, pub_qos_)),
  bbox_pub_(create_publisher<vision_msgs::msg::Detection2D>(out_bbox_topic_name_, pub_qos_)),
  pose_estimate_pub_(create_publisher<vision_msgs::msg::Detection3DArray>(
      out_pose_estimate_topic_name_, pub_qos_)),
  img_msg_(nullptr),
  cam_info_msg_(nullptr),
  depth_msg_(nullptr),
  pose_estimate_msg_(nullptr),
  action_cb_group_(create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive)),
  subscription_cb_group_(create_callback_group(rclcpp::CallbackGroupType::Reentrant))
{
  action_server_ = rclcpp_action::create_server<EstimatePoseFoundationPose>(
    this,
    action_name_,
    std::bind(
      &FoundationPoseServer::HandleGoal, this, std::placeholders::_1, std::placeholders::_2),
    std::bind(&FoundationPoseServer::HandleCancel, this, std::placeholders::_1),
    std::bind(&FoundationPoseServer::HandleAccepted, this, std::placeholders::_1),
    rcl_action_server_get_default_options(),
    action_cb_group_);

  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = subscription_cb_group_;

  img_sub_ = create_subscription<sensor_msgs::msg::Image>(
    in_img_topic_name_, sub_qos_,
    std::bind(&FoundationPoseServer::CallbackImg, this, std::placeholders::_1),
    sub_options);
  cam_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
    in_camera_info_topic_name_, sub_qos_,
    std::bind(&FoundationPoseServer::CallbackCameraInfo, this, std::placeholders::_1),
    sub_options);
  depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
    in_depth_topic_name_, sub_qos_,
    std::bind(&FoundationPoseServer::CallbackDepth, this, std::placeholders::_1),
    sub_options);
  pose_estimate_sub_ = create_subscription<vision_msgs::msg::Detection3DArray>(
    in_pose_estimate_topic_name_, sub_qos_,
    std::bind(&FoundationPoseServer::CallbackPoseEstimate, this, std::placeholders::_1),
    sub_options);
}

rclcpp_action::GoalResponse FoundationPoseServer::HandleGoal(
  const rclcpp_action::GoalUUID & uuid,
  std::shared_ptr<const EstimatePoseFoundationPose::Goal> goal)
{
  RCLCPP_INFO(get_logger(), "Received goal request for foundation pose estimation");
  (void)uuid;
  (void)goal;
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse FoundationPoseServer::HandleCancel(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<EstimatePoseFoundationPose>> goal_handle)
{
  RCLCPP_INFO(get_logger(), "Received request to cancel goal");
  (void)goal_handle;
  ClearAllMsgs();
  return rclcpp_action::CancelResponse::ACCEPT;
}

void FoundationPoseServer::HandleAccepted(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<EstimatePoseFoundationPose>> goal_handle)
{
  // this needs to return quickly to avoid blocking the executor, so spin up a new thread
  std::thread{std::bind(&FoundationPoseServer::Execute, this, std::placeholders::_1),
    goal_handle}.detach();
}

void FoundationPoseServer::Execute(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<EstimatePoseFoundationPose>> goal_handle)
{
  auto goal = goal_handle->get_goal();
  auto bbox_msg = goal->roi;
  auto header = std_msgs::msg::Header();
  header.stamp.sec = bbox_msg.header.stamp.sec;
  header.stamp.nanosec = bbox_msg.header.stamp.nanosec;
  RCLCPP_INFO(
    get_logger(), "ROI Header timestamp  {%d.%d}.", header.stamp.sec,
    bbox_msg.header.stamp.nanosec);

  auto result = std::make_shared<EstimatePoseFoundationPose::Result>();
  RCLCPP_INFO(get_logger(), "Executing goal");
  // Wait for the latest image.
  while ((img_msg_ == nullptr || cam_info_msg_ == nullptr || depth_msg_ == nullptr) &&
    goal_handle->is_executing() && (!goal_handle->is_canceling()))
  {
    RCLCPP_DEBUG_ONCE(get_logger(), "Waiting for all the messages to come");
  }

  if (goal_handle->is_canceling()) {
    goal_handle->canceled(result);
    RCLCPP_INFO(this->get_logger(), "Goal Canceled for action EstimatePoseFoundationPose");
    return;
  }

  if (img_msg_ != nullptr) {
    auto img_msg = *img_msg_;
    img_msg.header.stamp = header.stamp;
    // Publish the latest image
    img_pub_->publish(img_msg);
    // Reset the image for the next request.
    img_msg_.reset();
  }

  if (cam_info_msg_ != nullptr) {
    auto cam_info_msg = *cam_info_msg_;
    cam_info_msg.header.stamp = header.stamp;
    // Publish the latest camera_info
    cam_info_pub_->publish(cam_info_msg);
    // Reset the camera_info for the next request.
    cam_info_msg_.reset();
  }

  if (depth_msg_ != nullptr) {
    auto depth_msg = *depth_msg_;
    depth_msg.header.stamp = header.stamp;
    // Publish the latest depth image
    depth_pub_->publish(depth_msg);
    // Reset the depth image for the next request.
    depth_msg_.reset();
  }

  bbox_pub_->publish(bbox_msg);

  RCLCPP_INFO(get_logger(), "All msgs published, waiting for poses.");

  // Wait for the pose estimation result. Reset for every new request.
  while (pose_estimate_msg_ == nullptr && goal_handle->is_executing() &&
    (!goal_handle->is_canceling()))
  {
    RCLCPP_DEBUG_ONCE(get_logger(), "Waiting for pose estimation result");
  }

  if (goal_handle->is_canceling()) {
    goal_handle->canceled(result);
    RCLCPP_INFO(this->get_logger(), "Goal Canceled for action EstimatePoseFoundationPose");
    return;
  }

  if (pose_estimate_msg_ != nullptr) {
    RCLCPP_INFO(get_logger(), "Publishing pose estimation result");
    result->poses = *pose_estimate_msg_;
    pose_estimate_msg_.reset();

    goal_handle->succeed(result);
  }
}

void FoundationPoseServer::CallbackImg(const sensor_msgs::msg::Image::ConstSharedPtr & msg_ptr)
{
  img_msg_ = msg_ptr;
  RCLCPP_DEBUG_ONCE(get_logger(), "[CallbackImg] Received image");
}

void FoundationPoseServer::CallbackCameraInfo(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg_ptr)
{
  cam_info_msg_ = msg_ptr;
  RCLCPP_DEBUG_ONCE(get_logger(), "[CallbackCameraInfo] Received camera_info");
}

void FoundationPoseServer::CallbackDepth(const sensor_msgs::msg::Image::ConstSharedPtr & msg_ptr)
{
  depth_msg_ = msg_ptr;
  RCLCPP_DEBUG_ONCE(get_logger(), "[CallbackDepth] Received depth image");
}

void FoundationPoseServer::CallbackPoseEstimate(
  const vision_msgs::msg::Detection3DArray::ConstSharedPtr & msg_ptr)
{
  pose_estimate_msg_ = msg_ptr;
  RCLCPP_DEBUG_ONCE(get_logger(), "[CallbackPoseEstimate] Received poses");
}

void FoundationPoseServer::ClearAllMsgs()
{
  img_msg_.reset();
  cam_info_msg_.reset();
  depth_msg_.reset();
  pose_estimate_msg_.reset();
}

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac::manipulation::FoundationPoseServer)
