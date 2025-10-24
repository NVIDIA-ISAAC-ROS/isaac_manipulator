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

#include "isaac_manipulator_servers/segment_anything_server.hpp"
#include "vision_msgs/msg/detection2_d.hpp"

namespace
{
constexpr double kDetectionSize = 10.0;
const char kObjectID[] = "0";
}

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

SegmentAnythingServer::SegmentAnythingServer(const rclcpp::NodeOptions & options)
: Node("segment_anything_server", options),
  is_sam2_(declare_parameter<bool>("is_sam2", false)),
  action_name_(declare_parameter<std::string>("action_name", "segment_anything")),
  in_img_topic_name_(declare_parameter<std::string>("in_img_topic_name", "image_color")),
  out_img_topic_name_(declare_parameter<std::string>(
      "out_img_topic_name", "segment_anything_server/image_color")),
  in_camera_info_topic_name_(declare_parameter<std::string>(
      "in_camera_info_topic_name", "camera_info")),
  out_camera_info_topic_name_(declare_parameter<std::string>(
      "out_camera_info_topic_name", "segment_anything_server/camera_info")),
  in_segmentation_mask_topic_name_(declare_parameter<std::string>(
      "in_segmentation_mask_topic_name", "segmentation_mask")),
  in_detections_topic_name_(declare_parameter<std::string>(
      "in_detections_topic_name", "detections")),
  out_detections_topic_name_(declare_parameter<std::string>(
      "out_detections_topic_name", "segment_anything_server/detections_initial_guess")),
  service_call_timeout_{static_cast<int>(declare_parameter<int>(
      "service_call_timeout", 5))},
  service_discovery_timeout_{static_cast<int>(declare_parameter<int>(
      "service_discovery_timeout", 5))},
  input_qos_{::isaac_ros::common::AddQosParameter(
      *this, "SENSOR_DATA", "input_qos")},
  result_and_output_qos_{::isaac_ros::common::AddQosParameter(
      *this, "DEFAULT", "result_and_output_qos")},
  img_pub_(create_publisher<sensor_msgs::msg::Image>(out_img_topic_name_, result_and_output_qos_)),
  cam_info_pub_(
    create_publisher<sensor_msgs::msg::CameraInfo>(
      out_camera_info_topic_name_, result_and_output_qos_)),
  detections_pub_(create_publisher<vision_msgs::msg::Detection2DArray>(
      out_detections_topic_name_, result_and_output_qos_)),
  img_msg_(nullptr),
  cam_info_msg_(nullptr),
  detections_msg_(nullptr),
  action_cb_group_(create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive)),
  client_cb_group_(create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive)),
  subscription_cb_group_(create_callback_group(rclcpp::CallbackGroupType::Reentrant))
{
  action_server_ = rclcpp_action::create_server<SegmentAnything>(
    this,
    action_name_,
    std::bind(
      &SegmentAnythingServer::HandleGoal, this, std::placeholders::_1, std::placeholders::_2),
    std::bind(&SegmentAnythingServer::HandleCancel, this, std::placeholders::_1),
    std::bind(&SegmentAnythingServer::HandleAccepted, this, std::placeholders::_1),
    rcl_action_server_get_default_options(),
    action_cb_group_);

  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = subscription_cb_group_;

  img_sub_ = create_subscription<sensor_msgs::msg::Image>(
    in_img_topic_name_, input_qos_,
    std::bind(&SegmentAnythingServer::CallbackImg, this, std::placeholders::_1),
    sub_options);
  cam_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
    in_camera_info_topic_name_, input_qos_,
    std::bind(&SegmentAnythingServer::CallbackCameraInfo, this, std::placeholders::_1),
    sub_options);
  // We make this same as pub qos as that is input of the result of the computation.
  segmentation_mask_sub_ = create_subscription<sensor_msgs::msg::Image>(
    in_segmentation_mask_topic_name_, result_and_output_qos_,
    std::bind(&SegmentAnythingServer::CallbackSegmentationMask, this, std::placeholders::_1),
    sub_options);
  detections_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
    in_detections_topic_name_, result_and_output_qos_,
    std::bind(&SegmentAnythingServer::CallbackDetections, this, std::placeholders::_1),
    sub_options);

  add_objects_client_ = create_client<AddObjects>(
    "/segment_anything2/add_objects", rclcpp::ServicesQoS(), client_cb_group_);
  remove_object_client_ = create_client<RemoveObject>(
    "/segment_anything2/remove_object", rclcpp::ServicesQoS(), client_cb_group_);
}

rclcpp_action::GoalResponse SegmentAnythingServer::HandleGoal(
  const rclcpp_action::GoalUUID & uuid,
  std::shared_ptr<const SegmentAnything::Goal> goal)
{
  RCLCPP_INFO(get_logger(), "Received goal request for segment anything pose estimation");
  (void)uuid;
  (void)goal;
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse SegmentAnythingServer::HandleCancel(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<SegmentAnything>> goal_handle)
{
  RCLCPP_INFO(get_logger(), "Received request to cancel goal");
  (void)goal_handle;
  ClearAllMsgs();
  return rclcpp_action::CancelResponse::ACCEPT;
}

void SegmentAnythingServer::HandleAccepted(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<SegmentAnything>> goal_handle)
{
  std::thread{std::bind(&SegmentAnythingServer::Execute, this, std::placeholders::_1),
    goal_handle}.detach();
}

vision_msgs::msg::Detection2DArray SegmentAnythingServer::PointToDetection2DArray(
  const vision_msgs::msg::Point2D & point,
  const rclcpp::Time & timestamp)
{
  vision_msgs::msg::Detection2DArray detections_array;
  detections_array.header.stamp = timestamp;

  vision_msgs::msg::Detection2D detection;
  detection.header.stamp = timestamp;
  detection.bbox.center.position.x = point.x;
  detection.bbox.center.position.y = point.y;

  // Set size to a small value since it's a point click
  detection.bbox.size_x = kDetectionSize;
  detection.bbox.size_y = kDetectionSize;

  detections_array.detections.push_back(detection);

  return detections_array;
}

vision_msgs::msg::Detection2DArray SegmentAnythingServer::BoundingBox2DToDetection2DArray(
  const vision_msgs::msg::BoundingBox2D & bbox,
  const rclcpp::Time & timestamp)
{
  vision_msgs::msg::Detection2DArray detections_array;
  detections_array.header.stamp = timestamp;

  vision_msgs::msg::Detection2D detection;
  detection.header.stamp = timestamp;
  detection.bbox = bbox;

  detections_array.detections.push_back(detection);

  return detections_array;
}

void SegmentAnythingServer::Execute(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<SegmentAnything>> goal_handle)
{
  auto goal = goal_handle->get_goal();
  auto header = std_msgs::msg::Header();
  header.stamp = this->now();

  auto result = std::make_shared<SegmentAnything::Result>();
  RCLCPP_INFO(get_logger(), "Executing goal");

  // Wait for all required messages
  while ((img_msg_ == nullptr || cam_info_msg_ == nullptr) &&
    goal_handle->is_executing() && (!goal_handle->is_canceling()))
  {
    RCLCPP_DEBUG_ONCE(get_logger(), "Waiting for all the messages to come");
  }

  if (goal_handle->is_canceling()) {
    goal_handle->canceled(result);
    RCLCPP_INFO(this->get_logger(), "Goal Canceled for action SegmentAnything");
    return;
  }

  RCLCPP_INFO(get_logger(), "All messages received, publishing...");

  // Publish all messages with the same timestamp
  if (img_msg_ != nullptr) {
    auto img_msg = *img_msg_;
    img_msg.header.stamp = header.stamp;
    img_pub_->publish(img_msg);
    img_msg_.reset();
    RCLCPP_INFO(get_logger(), "Published image");
  }

  if (cam_info_msg_ != nullptr) {
    auto cam_info_msg = *cam_info_msg_;
    cam_info_msg.header.stamp = header.stamp;
    cam_info_pub_->publish(cam_info_msg);
    cam_info_msg_.reset();
    RCLCPP_INFO(get_logger(), "Published camera info");
  }

  if (is_sam2_) {
    // If using SAM2, call required services to segment the object
    if (!CallAddObjectsService(header, goal->use_point_hint,
      goal->initial_hint_bbox, goal->initial_hint_point))
    {
      goal_handle->abort(result);
      return;
    }
  } else {
    // If using SAM1, publish the point as a Detection2DArray with the same timestamp
    auto detections_msg = goal->use_point_hint ?
      PointToDetection2DArray(goal->initial_hint_point, header.stamp) :
      BoundingBox2DToDetection2DArray(goal->initial_hint_bbox, header.stamp);
    detections_pub_->publish(detections_msg);
    RCLCPP_INFO(get_logger(), "Published detections");
  }

  RCLCPP_INFO(get_logger(), "All messages published, waiting for segmentation mask.");

  while (segmentation_mask_msg_ == nullptr && goal_handle->is_executing() &&
    (!goal_handle->is_canceling()))
  {
    RCLCPP_DEBUG_ONCE(get_logger(), "Waiting for segmentation mask result");
  }

  if (goal_handle->is_canceling()) {
    goal_handle->canceled(result);
    RCLCPP_INFO(this->get_logger(), "Goal Canceled for action SegmentAnything");
    return;
  }

  RCLCPP_INFO(get_logger(), "Segmentation mask received, waiting for detections.");

  while (detections_msg_ == nullptr && goal_handle->is_executing() &&
    (!goal_handle->is_canceling()))
  {
    RCLCPP_DEBUG_ONCE(get_logger(), "Waiting for detections result");
  }

  if (goal_handle->is_canceling()) {
    goal_handle->canceled(result);
    RCLCPP_INFO(this->get_logger(), "Goal Canceled for action SegmentAnything");
    return;
  }

  RCLCPP_INFO(get_logger(), "Detections also received, publishing result.");

  if (detections_msg_ != nullptr && segmentation_mask_msg_ != nullptr) {
    // Assemble the detections and segmentation mask result
    // The below code creates a copy of the detections and segmentation mask
    // We take the first detection result for now since we do not support
    // initial hints in batch currently.
    auto local_detections = *detections_msg_;
    auto local_mask = *segmentation_mask_msg_;

    if (is_sam2_) {
      // If using SAM2, remove the added object
      if (!CallRemoveObjectService(header)) {
        goal_handle->abort(result);
        return;
      }
    }

    // Clear shared pointers early to prevent race conditions
    detections_msg_.reset();
    segmentation_mask_msg_.reset();

    RCLCPP_INFO(get_logger(), "Publishing detections result");

    if (local_detections.detections.empty()) {
      RCLCPP_ERROR(get_logger(), "No detections available");
      goal_handle->abort(result);
      return;
    }

    try {
      result->detection = local_detections.detections[0];
      result->segmentation_mask = local_mask;

      goal_handle->succeed(result);
      RCLCPP_INFO(get_logger(), "Successfully published result");
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Error assembling detections and segmentation mask: %s", e.what());
      goal_handle->abort(result);
      return;
    }
  } else {
    RCLCPP_ERROR(get_logger(), "No detections or segmentation mask received");
    goal_handle->abort(result);
    return;
  }
}

bool SegmentAnythingServer::CallAddObjectsService(
  std_msgs::msg::Header header, bool use_point_hint,
  vision_msgs::msg::BoundingBox2D initial_hint_bbox, vision_msgs::msg::Point2D initial_hint_point)
{
  // Wait for add objects service to be available
  if (!add_objects_client_->wait_for_service(std::chrono::seconds(service_discovery_timeout_))) {
    RCLCPP_ERROR(get_logger(), "AddObjects service not available");
    return false;
  }

  // Create a request for the AddObjects service. Object ID is arbitrarily
  // chosen to be 0 since we only segment one object at a time.
  auto request = std::make_shared<AddObjects::Request>();
  request->request_header = header;

  if (use_point_hint) {
    request->point_coords.push_back(initial_hint_point);
    request->point_object_ids.push_back(kObjectID);
    request->point_labels.push_back(1);  // 1 for foreground
  } else {
    request->bbox_coords.push_back(initial_hint_bbox);
    request->bbox_object_ids.push_back(kObjectID);
  }

  // Call the AddObjects service with the request
  auto future = add_objects_client_->async_send_request(request);
  if (future.wait_for(std::chrono::seconds(service_call_timeout_)) == std::future_status::timeout) {
    RCLCPP_ERROR(get_logger(), "AddObjects service call timed out");
    return false;
  }

  // Get the response from the AddObjects service
  auto response = future.get();
  if (!response->success) {
    RCLCPP_ERROR(get_logger(), "AddObjects service failed: %s", response->message.c_str());
    return false;
  }

  return true;
}

bool SegmentAnythingServer::CallRemoveObjectService(std_msgs::msg::Header header)
{
  // Wait for remove object service to be available
  if (!remove_object_client_->wait_for_service(std::chrono::seconds(service_discovery_timeout_))) {
    RCLCPP_ERROR(get_logger(), "RemoveObject service not available");
    return false;
  }

  // Create a request for the RemoveObject service
  auto request = std::make_shared<RemoveObject::Request>();
  request->request_header = header;
  request->object_id = kObjectID;

  // Call the RemoveObject service
  auto future = remove_object_client_->async_send_request(request);
  if (future.wait_for(std::chrono::seconds(service_call_timeout_)) == std::future_status::timeout) {
    RCLCPP_ERROR(get_logger(), "RemoveObject service call timed out");
    return false;
  }

  // Get the response from the RemoveObject service
  auto response = future.get();
  if (!response->success) {
    RCLCPP_ERROR(get_logger(), "RemoveObject service failed: %s", response->message.c_str());
    return false;
  }

  return true;
}

void SegmentAnythingServer::CallbackImg(const sensor_msgs::msg::Image::ConstSharedPtr & msg_ptr)
{
  img_msg_ = msg_ptr;
  RCLCPP_DEBUG_ONCE(get_logger(), "[CallbackImg] Received image");
}

void SegmentAnythingServer::CallbackCameraInfo(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg_ptr)
{
  cam_info_msg_ = msg_ptr;
  RCLCPP_DEBUG_ONCE(get_logger(), "[CallbackCameraInfo] Received camera_info");
}

void SegmentAnythingServer::CallbackSegmentationMask(
  const sensor_msgs::msg::Image::ConstSharedPtr & msg_ptr)
{
  segmentation_mask_msg_ = msg_ptr;
  RCLCPP_DEBUG_ONCE(get_logger(), "[CallbackSegmentationMask] Received segmentation mask");
}

void SegmentAnythingServer::CallbackDetections(
  const vision_msgs::msg::Detection2DArray::ConstSharedPtr & msg_ptr)
{
  detections_msg_ = msg_ptr;
  RCLCPP_DEBUG_ONCE(get_logger(), "[CallbackDetections] Received detections");
}

void SegmentAnythingServer::ClearAllMsgs()
{
  img_msg_.reset();
  cam_info_msg_.reset();
  segmentation_mask_msg_.reset();
  detections_msg_.reset();
}

SegmentAnythingServer::~SegmentAnythingServer()
{
  ClearAllMsgs();
}

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac::manipulation::SegmentAnythingServer)
