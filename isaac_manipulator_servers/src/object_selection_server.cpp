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

#include "isaac_manipulator_servers/object_selection_server.hpp"
#include <algorithm>
#include <functional>

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

ObjectSelectionServer::ObjectSelectionServer(const rclcpp::NodeOptions & options)
: Node("object_selection_server", options),
  action_name_(declare_parameter<std::string>("action_name", "get_selected_object")),
  action_cb_group_(create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive)),
  rng_(std::random_device{}())
{
  std::string policy_param = declare_parameter<std::string>("selection_policy", "first");
  // Convert to lower case for case insensitive comparison
  std::string policy_param_lower = policy_param;
  std::transform(
    policy_param_lower.begin(), policy_param_lower.end(),
    policy_param_lower.begin(), ::tolower);
  if (policy_param_lower == "highest_score") {
    policy_ = SelectionPolicy::HIGHEST_SCORE;
  } else if (policy_param_lower == "first") {
    policy_ = SelectionPolicy::FIRST;
  } else if (policy_param_lower == "random") {
    policy_ = SelectionPolicy::RANDOM;
  } else {
    RCLCPP_ERROR(get_logger(), "Unknown selection_policy '%s'", policy_param.c_str());
    throw std::invalid_argument("Unknown selection policy");
  }

  // Register parameter change callback
  parameter_callback_handle_ = this->add_on_set_parameters_callback(
    std::bind(&ObjectSelectionServer::OnParameterChange, this, std::placeholders::_1));

  action_server_ = rclcpp_action::create_server<GetSelectedObject>(
    this,
    action_name_,
    std::bind(
      &ObjectSelectionServer::HandleGoal, this, std::placeholders::_1,
      std::placeholders::_2),
    std::bind(&ObjectSelectionServer::HandleCancel, this, std::placeholders::_1),
    std::bind(&ObjectSelectionServer::HandleAccepted, this, std::placeholders::_1),
    rcl_action_server_get_default_options(),
    action_cb_group_);
}

rclcpp_action::GoalResponse ObjectSelectionServer::HandleGoal(
  const rclcpp_action::GoalUUID & uuid, std::shared_ptr<const GetSelectedObject::Goal> goal)
{
  RCLCPP_INFO(get_logger(), "Received goal request for object selection");
  (void)uuid;
  if (!goal || goal->detections.detections.empty()) {
    RCLCPP_WARN(get_logger(), "Goal contains no detections");
    return rclcpp_action::GoalResponse::REJECT;
  }
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse ObjectSelectionServer::HandleCancel(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<GetSelectedObject>> goal_handle)
{
  RCLCPP_INFO(get_logger(), "Received request to cancel goal");
  (void)goal_handle;
  return rclcpp_action::CancelResponse::ACCEPT;
}

void ObjectSelectionServer::HandleAccepted(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<GetSelectedObject>> goal_handle)
{
  std::thread{std::bind(&ObjectSelectionServer::Execute, this, std::placeholders::_1),
    goal_handle}.detach();
}

void ObjectSelectionServer::Execute(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<GetSelectedObject>> goal_handle)
{
  RCLCPP_INFO(get_logger(), "Executing object selection goal");
  const auto goal = goal_handle->get_goal();

  // Select the detection with the selected policy
  auto selected = SelectDetection(goal->detections);

  auto result = std::make_shared<GetSelectedObject::Result>();
  result->selected_detection = selected;
  goal_handle->succeed(result);
  RCLCPP_INFO(get_logger(), "Object selection succeeded");
}

vision_msgs::msg::Detection2D ObjectSelectionServer::SelectDetection(
  const vision_msgs::msg::Detection2DArray & detections)
{
  switch (policy_) {
    case SelectionPolicy::HIGHEST_SCORE:
      return SelectHighestScore(detections);
    case SelectionPolicy::FIRST:
      return SelectFirst(detections);
    case SelectionPolicy::RANDOM:
      return SelectRandom(detections);
    default:
      RCLCPP_ERROR(get_logger(), "Unknown selection policy in SelectDetection!");
      throw std::invalid_argument("Unknown selection policy in SelectDetection");
  }
}

vision_msgs::msg::Detection2D ObjectSelectionServer::SelectHighestScore(
  const vision_msgs::msg::Detection2DArray & detections)
{
  if (detections.detections.empty()) {
    return vision_msgs::msg::Detection2D();
  }
  auto best_it = detections.detections.begin();
  float best_score = -1.0f;
  for (auto it = detections.detections.begin(); it != detections.detections.end(); ++it) {
    if (!it->results.empty() && it->results[0].hypothesis.score > best_score) {
      best_score = it->results[0].hypothesis.score;
      best_it = it;
    }
  }
  return *best_it;
}

vision_msgs::msg::Detection2D ObjectSelectionServer::SelectFirst(
  const vision_msgs::msg::Detection2DArray & detections)
{
  if (detections.detections.empty()) {
    return vision_msgs::msg::Detection2D();
  }
  return detections.detections.front();
}

vision_msgs::msg::Detection2D ObjectSelectionServer::SelectRandom(
  const vision_msgs::msg::Detection2DArray & detections)
{
  if (detections.detections.empty()) {
    return vision_msgs::msg::Detection2D();
  }
  std::uniform_int_distribution<size_t> dist(0, detections.detections.size() - 1);
  return detections.detections[dist(rng_)];
}

rcl_interfaces::msg::SetParametersResult ObjectSelectionServer::OnParameterChange(
  const std::vector<rclcpp::Parameter> & params)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "";
  for (const auto & param : params) {
    if (param.get_name() == "selection_policy") {
      std::string policy_param_lower = param.as_string();
      std::transform(
        policy_param_lower.begin(), policy_param_lower.end(),
        policy_param_lower.begin(), ::tolower);
      if (policy_param_lower == "highest_score") {
        policy_ = SelectionPolicy::HIGHEST_SCORE;
        RCLCPP_INFO(get_logger(), "selection_policy changed to 'highest_score'");
      } else if (policy_param_lower == "first") {
        policy_ = SelectionPolicy::FIRST;
        RCLCPP_INFO(get_logger(), "selection_policy changed to 'first'");
      } else if (policy_param_lower == "random") {
        policy_ = SelectionPolicy::RANDOM;
        RCLCPP_INFO(get_logger(), "selection_policy changed to 'random'");
      } else {
        result.successful = false;
        result.reason = "Invalid value for selection_policy parameter";
        RCLCPP_ERROR(
          get_logger(), "Invalid value for selection_policy: '%s'",
          param.as_string().c_str());
      }
    }
  }
  return result;
}

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac::manipulation::ObjectSelectionServer)
