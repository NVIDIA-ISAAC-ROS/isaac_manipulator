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

#include <chrono>

#include "isaac_manipulator_servers/foundation_pose_server.hpp"


namespace
{
const std::chrono::nanoseconds PARAMETER_SERVICE_TIMEOUT = std::chrono::seconds(1);
bool ServiceAvailable(const rclcpp::AsyncParametersClient::SharedPtr & parameter_client)
{
  return parameter_client->wait_for_service(PARAMETER_SERVICE_TIMEOUT);
}
}  // namespace

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
  out_segmented_mask_topic_name_(declare_parameter<std::string>(
      "out_segmented_mask_topic_name", "foundation_pose_server/segmented_mask")),
  foundation_pose_node_name_(declare_parameter<std::string>(
      "foundation_pose_node_name", "/foundationpose_node")),
  input_qos_{::isaac_ros::common::AddQosParameter(*this, "SENSOR_DATA", "input_qos")},
  result_and_output_qos_{::isaac_ros::common::AddQosParameter(
      *this, "DEFAULT", "result_and_output_qos")},
  img_pub_(create_publisher<sensor_msgs::msg::Image>(
      out_img_topic_name_, result_and_output_qos_)),
  cam_info_pub_(
    create_publisher<sensor_msgs::msg::CameraInfo>(
      out_camera_info_topic_name_, result_and_output_qos_)),
  depth_pub_(create_publisher<sensor_msgs::msg::Image>(
      out_depth_topic_name_, result_and_output_qos_)),
  bbox_pub_(create_publisher<vision_msgs::msg::Detection2D>(
      out_bbox_topic_name_, result_and_output_qos_)),
  pose_estimate_pub_(create_publisher<vision_msgs::msg::Detection3DArray>(
      out_pose_estimate_topic_name_, result_and_output_qos_)),
  segmented_mask_pub_(create_publisher<sensor_msgs::msg::Image>(
      out_segmented_mask_topic_name_, result_and_output_qos_)),
  img_msg_(nullptr),
  cam_info_msg_(nullptr),
  depth_msg_(nullptr),
  pose_estimate_msg_(nullptr),
  action_cb_group_(create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive)),
  subscription_cb_group_(create_callback_group(rclcpp::CallbackGroupType::Reentrant)),
  parameter_client_(std::make_shared<rclcpp::AsyncParametersClient>(this,
        foundation_pose_node_name_))
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
    in_img_topic_name_, input_qos_,
    std::bind(&FoundationPoseServer::CallbackImg, this, std::placeholders::_1),
    sub_options);
  cam_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
    in_camera_info_topic_name_, input_qos_,
    std::bind(&FoundationPoseServer::CallbackCameraInfo, this, std::placeholders::_1),
    sub_options);
  depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
    in_depth_topic_name_, input_qos_,
    std::bind(&FoundationPoseServer::CallbackDepth, this, std::placeholders::_1),
    sub_options);
  // We make this same as pub qos as that is input of the result of the computation.
  pose_estimate_sub_ = create_subscription<vision_msgs::msg::Detection3DArray>(
    in_pose_estimate_topic_name_, result_and_output_qos_,
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
  auto segmentation_mask_msg = goal->segmentation_mask;
  auto header = std_msgs::msg::Header();
  // This is important that the header contains latest timestamp as the foundation pose will
  // discard input that is older than X seconds to retain real time functionality.
  header.stamp = this->now();

  RCLCPP_INFO(
    get_logger(), "ROI Header timestamp  {%d.%d}.", header.stamp.sec, header.stamp.nanosec);

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

  std::vector<rclcpp::Parameter> parameters;
  if (!goal->mesh_file_path.empty()) {
    parameters.push_back(rclcpp::Parameter("mesh_file_path", goal->mesh_file_path));
  }
  if (!goal->object_frame_name.empty()) {
    parameters.push_back(rclcpp::Parameter("tf_frame_name", goal->object_frame_name));
  }

  if (ServiceAvailable(parameter_client_) && !parameters.empty()) {
    auto callback = [this, goal_handle, result]
      (std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>>
      future) {
        for (const auto & set_parameters_result : future.get()) {
          if (!set_parameters_result.successful) {
            RCLCPP_ERROR(this->get_logger(),
                  "Failed to set parameters for foundation pose node. Reason: %s",
                  set_parameters_result.reason.c_str());
            goal_handle->abort(result);
            return;
          }
        }
        RCLCPP_INFO(this->get_logger(), "Parameters set successfully for foundation pose node.");
      };
    // Trigger FoundationPose parameter service to set the mesh resource
    auto set_parameters_future = parameter_client_->set_parameters(parameters, callback);
    // Wait for the parameters to be set
    set_parameters_future.wait_for(PARAMETER_SERVICE_TIMEOUT);
  }

  if (img_msg_ != nullptr) {
    auto img_msg = *img_msg_;
    img_msg.header.stamp = header.stamp;
    // Publish the latest image
    img_pub_->publish(img_msg);
    // Reset the image for the next request.
    img_msg_.reset();
    RCLCPP_INFO(get_logger(), "FP: Published image");
  } else {
    RCLCPP_ERROR(get_logger(), "FP: Image is null");
  }

  if (cam_info_msg_ != nullptr) {
    auto cam_info_msg = *cam_info_msg_;
    cam_info_msg.header.stamp = header.stamp;
    // Publish the latest camera_info
    cam_info_pub_->publish(cam_info_msg);
    // Reset the camera_info for the next request.
    cam_info_msg_.reset();
    RCLCPP_INFO(get_logger(), "FP: Published camera info");
  } else {
    RCLCPP_ERROR(get_logger(), "FP: Camera info is null");
  }

  if (depth_msg_ != nullptr) {
    auto depth_msg = *depth_msg_;
    depth_msg.header.stamp = header.stamp;
    // Publish the latest depth image
    depth_pub_->publish(depth_msg);
    // Reset the depth image for the next request.
    depth_msg_.reset();
    RCLCPP_INFO(get_logger(), "FP: Published depth image");
  } else {
    RCLCPP_ERROR(get_logger(), "FP: Depth image is null");
  }

  if (goal->use_segmentation_mask) {
    segmentation_mask_msg.header.stamp = header.stamp;
    segmented_mask_pub_->publish(segmentation_mask_msg);
    RCLCPP_INFO(get_logger(), "Published segmentation mask");
  } else {
    bbox_msg.header.stamp = header.stamp;
    bbox_pub_->publish(bbox_msg);
    RCLCPP_INFO(get_logger(), "Published bbox");
  }

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
  RCLCPP_DEBUG(get_logger(), "[CallbackPoseEstimate] Received poses");
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
