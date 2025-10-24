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

#ifndef ISAAC_MANIPULATOR_SERVERS__OBJECT_INFO_SERVER_HPP_
#define ISAAC_MANIPULATOR_SERVERS__OBJECT_INFO_SERVER_HPP_

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

#include "isaac_manipulator_interfaces/action/add_segmentation_mask.hpp"
#include "isaac_manipulator_interfaces/action/detect_objects.hpp"
#include "isaac_manipulator_interfaces/action/estimate_pose_dope.hpp"
#include "isaac_manipulator_interfaces/action/estimate_pose_foundation_pose.hpp"
#include "isaac_manipulator_interfaces/action/get_objects.hpp"
#include "isaac_manipulator_interfaces/action/get_object_pose.hpp"
#include "isaac_manipulator_interfaces/action/segment_anything.hpp"
#include "isaac_manipulator_interfaces/srv/add_mesh_to_object.hpp"
#include "isaac_manipulator_interfaces/srv/assign_name_to_object.hpp"
#include "isaac_manipulator_interfaces/srv/clear_objects.hpp"
#include "isaac_manipulator_servers/impl/action_clients.hpp"
#include "isaac_manipulator_servers/impl/action_client_manager.hpp"
#include "isaac_manipulator_servers/impl/backend_manager.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "vision_msgs/msg/bounding_box2_d.hpp"
#include "vision_msgs/msg/point2_d.hpp"

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

using GetObjectPoseAction = isaac_manipulator_interfaces::action::GetObjectPose;
using GetObjectsAction = isaac_manipulator_interfaces::action::GetObjects;
using AddSegmentationMaskAction = isaac_manipulator_interfaces::action::AddSegmentationMask;

using AddMeshToObjectSrv = isaac_manipulator_interfaces::srv::AddMeshToObject;
using ClearObjectsSrv = isaac_manipulator_interfaces::srv::ClearObjects;
using AssignNameToObjectSrv = isaac_manipulator_interfaces::srv::AssignNameToObject;

class ObjectInfoServer : public rclcpp::Node
{
public:
  explicit ObjectInfoServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~ObjectInfoServer() = default;

private:
  template<typename ActionTypeGoal>
  rclcpp_action::GoalResponse HandleGoal(
    const rclcpp_action::GoalUUID & uuid, std::shared_ptr<const ActionTypeGoal> goal);
  template<typename ActionType>
  rclcpp_action::CancelResponse HandleCancel(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<ActionType>> goal_handle);
  template<typename ActionType>
  void HandleAccepted(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<ActionType>> goal_handle);
  template<typename ActionType>
  void Execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<ActionType>> goal_handle);

  void ParameterCallback(const rcl_interfaces::msg::ParameterEvent & event);

  void RegisterActionClients(PoseEstimationBackend backend);

  void RegisterActionClients(ObjectDetectionBackend backend);

  void RegisterActionClients(SegmentationBackend backend);

  template<typename ActionType, typename BackendType>
  std::optional<typename ActionType::Result::SharedPtr> Trigger(
    const typename ActionType::Goal & goal);

  void ClearObjects(
    const std::shared_ptr<ClearObjectsSrv::Request> request,
    std::shared_ptr<ClearObjectsSrv::Response> response);

  void AddMeshToObject(
    const std::shared_ptr<AddMeshToObjectSrv::Request> request,
    std::shared_ptr<AddMeshToObjectSrv::Response> response);

  void AssignNameToObject(
    const std::shared_ptr<AssignNameToObjectSrv::Request> request,
    std::shared_ptr<AssignNameToObjectSrv::Response> response);

private:
  std::mutex objects_mutex_;

  rclcpp_action::Server<GetObjectsAction>::SharedPtr get_objects_server_;
  rclcpp_action::Server<GetObjectPoseAction>::SharedPtr object_info_server_;
  rclcpp_action::Server<AddSegmentationMaskAction>::SharedPtr segmentation_server_;
  rclcpp::Service<ClearObjectsSrv>::SharedPtr clear_objects_;
  rclcpp::Service<AddMeshToObjectSrv>::SharedPtr add_mesh_objects_;
  rclcpp::Service<AssignNameToObjectSrv>::SharedPtr assign_name_to_object_;
  std::unordered_map<int, isaac_manipulator_interfaces::msg::ObjectInfo> objects_;

  std::shared_ptr<ActionClientManager> action_client_manager_;

  std::string pose_estimation_backend_;
  std::string object_detection_backend_;
  std::string segmentation_backend_;
  std::shared_ptr<BackendManager> backend_manager_;

  std::shared_ptr<rclcpp::ParameterEventHandler> param_event_handler_;
  std::shared_ptr<rclcpp::ParameterEventCallbackHandle> param_event_cb_;

  rclcpp::CallbackGroup::SharedPtr get_objects_cb_group_;
  rclcpp::CallbackGroup::SharedPtr get_objects_pose_cb_group_;
  rclcpp::CallbackGroup::SharedPtr estimate_pose_fp_cb_group_;
  rclcpp::CallbackGroup::SharedPtr estimate_pose_dope_cb_group_;
  rclcpp::CallbackGroup::SharedPtr detect_objects_cb_group_;
  rclcpp::CallbackGroup::SharedPtr segmentation_cb_group_;
  rclcpp::CallbackGroup::SharedPtr clear_objects_cb_group_;
  rclcpp::CallbackGroup::SharedPtr add_mesh_objects_cb_group_;
  rclcpp::CallbackGroup::SharedPtr assign_name_to_object_cb_group_;
};

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

#endif  // ISAAC_MANIPULATOR_SERVERS__OBJECT_INFO_SERVER_HPP_
