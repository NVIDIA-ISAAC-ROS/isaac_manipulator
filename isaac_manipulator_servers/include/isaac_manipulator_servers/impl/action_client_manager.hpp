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

#ifndef ISAAC_MANIPULATOR_SERVERS__IMPL__ACTION_CLIENT_MANAGER_HPP_
#define ISAAC_MANIPULATOR_SERVERS__IMPL__ACTION_CLIENT_MANAGER_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

#include "isaac_manipulator_servers/impl/action_clients.hpp"
#include "isaac_manipulator_servers/impl/backend_types.hpp"
#include "isaac_manipulator_interfaces/action/add_segmentation_mask.hpp"
#include "isaac_manipulator_interfaces/action/detect_objects.hpp"
#include "isaac_manipulator_interfaces/action/estimate_pose_dope.hpp"
#include "isaac_manipulator_interfaces/action/estimate_pose_foundation_pose.hpp"
#include "isaac_manipulator_interfaces/action/segment_anything.hpp"

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

using AddSegmentationMaskAction = isaac_manipulator_interfaces::action::AddSegmentationMask;
using DetectObjectsAction = isaac_manipulator_interfaces::action::DetectObjects;
using DopeAction = isaac_manipulator_interfaces::action::EstimatePoseDope;
using FoundationPoseAction = isaac_manipulator_interfaces::action::EstimatePoseFoundationPose;
using SegmentAnythingAction = isaac_manipulator_interfaces::action::SegmentAnything;

template<typename ActionType>
using ActionClientPtr = std::shared_ptr<ActionClient<ActionType>>;

using action_client_variants = std::variant<ActionClientPtr<DetectObjectsAction>,
    ActionClientPtr<DopeAction>, ActionClientPtr<FoundationPoseAction>,
    ActionClientPtr<SegmentAnythingAction>>;

class ActionClientManager
{
public:
  ActionClientManager() = default;
  ~ActionClientManager() = default;

  template<typename BackendType, typename ActionType>
  void RegisterActionClient(const ActionClientPtr<ActionType> action_client);

  template<typename BackendType, typename ActionType>
  ActionClientPtr<ActionType> GetRegisteredClient() const;

private:
  std::unordered_map<std::string, action_client_variants> action_client_map_;
};

template<typename BackendType, typename ActionType>
void ActionClientManager::RegisterActionClient(ActionClientPtr<ActionType> action_client)
{
  std::string key = typeid(BackendType).name();
  action_client_map_[key] = action_client_variants(action_client);
}

template<typename BackendType, typename ActionType>
ActionClientPtr<ActionType> ActionClientManager::GetRegisteredClient() const
{
  std::string key = typeid(BackendType).name();
  return std::get<ActionClientPtr<ActionType>>(action_client_map_.at(key));
}

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

#endif  // ISAAC_MANIPULATOR_SERVERS__IMPL__ACTION_CLIENT_MANAGER_HPP_
