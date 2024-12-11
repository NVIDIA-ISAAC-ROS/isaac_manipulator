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

#ifndef ISAAC_MANIPULATOR_SERVERS__IMPL__ACTION_CLIENTS_HPP_
#define ISAAC_MANIPULATOR_SERVERS__IMPL__ACTION_CLIENTS_HPP_

#include <future>
#include <string>

#include "rclcpp_action/rclcpp_action.hpp"

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

// Forward declaration
class ObjectInfoServer;

template<typename ActionType>
class ActionClient
{
public:
  using GoalHandle = rclcpp_action::ClientGoalHandle<ActionType>;
  using ClientPtr = typename rclcpp_action::Client<ActionType>::SharedPtr;
  // One should be aware that allowing nullptr into a callback group will make the action client
  // use the same callback group as the Node it is created from potentially leading to deadlock
  // scenarios if there are multiple entities that require callbacks.
  explicit ActionClient(
    ObjectInfoServer & node, std::string action_name,
    rclcpp::CallbackGroup::SharedPtr callback_group = nullptr);
  ~ActionClient() = default;
  ClientPtr GetClient() const {return client_;}
  std::string GetActionName() const {return action_name_;}
  typename ActionType::Goal GetGoal();

private:
  void GoalResponseCallback(const typename GoalHandle::SharedPtr & goal_handle);
  void FeedbackCallback(
    typename GoalHandle::SharedPtr goal_handle,
    typename ActionType::Feedback::ConstSharedPtr feedback);
  void ResultCallback(const typename GoalHandle::WrappedResult & result);

private:
  ClientPtr client_;
  std::string action_name_;
};

template<typename ActionType>
ActionClient<ActionType>::ActionClient(
  ObjectInfoServer & node, std::string action_name,
  rclcpp::CallbackGroup::SharedPtr callback_group)
: action_name_(action_name)
{
  client_ = rclcpp_action::create_client<ActionType>(&node, action_name_, callback_group);
  auto send_goal_options = typename rclcpp_action::Client<ActionType>::SendGoalOptions();
  send_goal_options.goal_response_callback =
    std::bind(&ActionClient<ActionType>::GoalResponseCallback, this, std::placeholders::_1);
  send_goal_options.feedback_callback =
    std::bind(
    &ActionClient<ActionType>::FeedbackCallback, this, std::placeholders::_1,
    std::placeholders::_2);
  send_goal_options.result_callback =
    std::bind(&ActionClient<ActionType>::ResultCallback, this, std::placeholders::_1);
}

template<typename ActionType>
void ActionClient<ActionType>::GoalResponseCallback(
  const typename GoalHandle::SharedPtr & goal_handle)
{
  if (!goal_handle) {
    RCLCPP_ERROR(rclcpp::get_logger(action_name_), "Goal was rejected by the server");
  } else {
    RCLCPP_INFO(rclcpp::get_logger(action_name_), "Goal accepted by server, waiting for result");
  }
}

template<typename ActionType>
void ActionClient<ActionType>::FeedbackCallback(
  typename GoalHandle::SharedPtr goal_handle,
  typename ActionType::Feedback::ConstSharedPtr feedback)
{
  (void)goal_handle;
  RCLCPP_INFO(rclcpp::get_logger(action_name_), "Progress = %d", feedback->progress);
}

template<typename ActionType>
void ActionClient<ActionType>::ResultCallback(const typename GoalHandle::WrappedResult & result)
{
  RCLCPP_INFO(rclcpp::get_logger(action_name_), "Received result");
  switch (result.code) {
    case rclcpp_action::ResultCode::SUCCEEDED:
      break;
    case rclcpp_action::ResultCode::ABORTED:
      RCLCPP_ERROR(rclcpp::get_logger(action_name_), "Goal was aborted");
      return;
    case rclcpp_action::ResultCode::CANCELED:
      RCLCPP_ERROR(rclcpp::get_logger(action_name_), "Goal was canceled");
      return;
    default:
      RCLCPP_ERROR(rclcpp::get_logger(action_name_), "Unknown result code");
      return;
  }
}

template<typename ActionType>
typename ActionType::Goal ActionClient<ActionType>::GetGoal() {return typename ActionType::Goal();}


}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

#endif  // ISAAC_MANIPULATOR_SERVERS__IMPL__ACTION_CLIENTS_HPP_
