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

#ifndef ISAAC_MANIPULATOR_SERVERS__DOPE_SERVER_HPP_
#define ISAAC_MANIPULATOR_SERVERS__DOPE_SERVER_HPP_

#include <memory>
#include <string>

#include "isaac_manipulator_interfaces/action/estimate_pose_dope.hpp"
#include "isaac_ros_common/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

// The DopeServer class is a ROS 2 action server that receives an image and camera info and returns
// the detected poses of objects in the image. It creates the frontend for a DOPE ROS2 graph that
// performs the object pose estimation.
//
// Optionally, it can also publish the detected 2D bounding boxes of objects in the image when
// enable_2d_detections_ is set to true.

class DopeServer : public rclcpp::Node
{
public:
  explicit DopeServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~DopeServer() = default;

private:
  using EstimatePoseDope = isaac_manipulator_interfaces::action::EstimatePoseDope;

  rclcpp_action::GoalResponse HandleGoal(
    const rclcpp_action::GoalUUID & uuid, std::shared_ptr<const EstimatePoseDope::Goal> goal);
  rclcpp_action::CancelResponse HandleCancel(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<EstimatePoseDope>> goal_handle);
  void HandleAccepted(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<EstimatePoseDope>> goal_handle);
  void Execute(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<EstimatePoseDope>> goal_handle);

  void CallbackImg(const sensor_msgs::msg::Image::ConstSharedPtr & msg_ptr);
  void CallbackCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg_ptr);
  void CallbackPoseEstimate(const vision_msgs::msg::Detection3DArray::ConstSharedPtr & msg_ptr);
  void CallbackDetections(const vision_msgs::msg::Detection2DArray::ConstSharedPtr & msg_ptr);

private:
  std::string action_name_;
  std::string in_img_topic_name_;
  std::string out_img_topic_name_;
  std::string in_camera_info_topic_name_;
  std::string out_camera_info_topic_name_;
  std::string in_pose_estimate_topic_name_;
  std::string out_pose_estimate_topic_name_;
  std::string in_detections_topic_name_;
  std::string out_detections_topic_name_;

  bool enable_2d_detections_ = false;

  // QOS for subscriptions and publishers
  rclcpp::QoS sub_qos_;
  rclcpp::QoS pub_qos_;

  // ROS2 subscriptions and publishers
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr pose_estimate_sub_;
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detections_sub_;

  const rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub_;
  const rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_pub_;
  const rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr pose_estimate_pub_;
  const rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detections_pub_;

  // Messages to store the latest received data
  sensor_msgs::msg::Image::ConstSharedPtr img_msg_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr cam_info_msg_;
  vision_msgs::msg::Detection3DArray::ConstSharedPtr pose_estimate_msg_;
  vision_msgs::msg::Detection2DArray::ConstSharedPtr detections_msg_;

  rclcpp::CallbackGroup::SharedPtr action_cb_group_;
  rclcpp::CallbackGroup::SharedPtr subscription_cb_group_;

  rclcpp_action::Server<EstimatePoseDope>::SharedPtr action_server_;
};

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

#endif  // ISAAC_MANIPULATOR_SERVERS__DOPE_SERVER_HPP_
