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

#ifndef ISAAC_MANIPULATOR_SERVERS__OBJECT_SELECTION_SERVER_HPP_
#define ISAAC_MANIPULATOR_SERVERS__OBJECT_SELECTION_SERVER_HPP_

#include <memory>
#include <string>
#include <vector>
#include <random>

#include "isaac_manipulator_interfaces/action/get_selected_object.hpp"
#include "isaac_ros_common/qos.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "vision_msgs/msg/detection2_d.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include <rcl_interfaces/msg/set_parameters_result.hpp>

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

// The ObjectSelectionServer class is a ROS 2 action server that receives a list of Detection2D
// and selects one from the list based on a selection strategy (e.g., first, highest score, etc.).
class ObjectSelectionServer : public rclcpp::Node
{
public:
  enum class SelectionPolicy
  {
    HIGHEST_SCORE,
    FIRST,
    RANDOM
  };

  explicit ObjectSelectionServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~ObjectSelectionServer() = default;

private:
  using GetSelectedObject = isaac_manipulator_interfaces::action::GetSelectedObject;

  rclcpp_action::GoalResponse HandleGoal(
    const rclcpp_action::GoalUUID & uuid, std::shared_ptr<const GetSelectedObject::Goal> goal);
  rclcpp_action::CancelResponse HandleCancel(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<GetSelectedObject>> goal_handle);
  void HandleAccepted(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<GetSelectedObject>> goal_handle);
  void Execute(
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<GetSelectedObject>> goal_handle);

  // Selection strategies
  vision_msgs::msg::Detection2D SelectDetection(
    const vision_msgs::msg::Detection2DArray & detections);
  vision_msgs::msg::Detection2D SelectHighestScore(
    const vision_msgs::msg::Detection2DArray & detections);
  vision_msgs::msg::Detection2D SelectFirst(const vision_msgs::msg::Detection2DArray & detections);
  vision_msgs::msg::Detection2D SelectRandom(const vision_msgs::msg::Detection2DArray & detections);

  // Parameter change callback
  rcl_interfaces::msg::SetParametersResult OnParameterChange(
    const std::vector<rclcpp::Parameter> & params);
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;

  SelectionPolicy policy_;
  std::string action_name_;
  rclcpp::CallbackGroup::SharedPtr action_cb_group_;
  rclcpp_action::Server<GetSelectedObject>::SharedPtr action_server_;
  std::mt19937 rng_;
};

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

#endif  // ISAAC_MANIPULATOR_SERVERS__OBJECT_SELECTION_SERVER_HPP_
