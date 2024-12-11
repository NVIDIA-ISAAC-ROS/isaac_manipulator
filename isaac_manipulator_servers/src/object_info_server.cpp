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
#include <cmath>
#include <vector>

#include "isaac_manipulator_servers/object_info_server.hpp"
#include "isaac_manipulator_servers/impl/action_clients.hpp"

namespace
{
using PoseEstimationBackend = nvidia::isaac::manipulation::PoseEstimationBackend;
using ObjectDetectionBackend = nvidia::isaac::manipulation::ObjectDetectionBackend;
// User string to pose estimation backend mode
const std::unordered_map<std::string, PoseEstimationBackend> POSE_ESTIMATION_BACKEND({
    {"FOUNDATION_POSE", PoseEstimationBackend::FOUNDATION_POSE},
    {"DOPE", PoseEstimationBackend::DOPE}
  });

// User string to object detection backend mode
const std::unordered_map<std::string, ObjectDetectionBackend> OBJECT_DETECTION_BACKEND({
    {"RT_DETR", ObjectDetectionBackend::RT_DETR},
    {"DOPE", ObjectDetectionBackend::DOPE}
  });

template<typename BackendType>
std::optional<BackendType> ValidateBackend(
  const std::string & backend, const std::unordered_map<std::string, BackendType> & backend_map)
{
  const auto backend_it = backend_map.find(backend);
  try {
    if (backend_it == std::end(backend_map)) {
      RCLCPP_WARN(
        rclcpp::get_logger("ValidateBackend"), "Unsupported backend: [%s]", backend.c_str());
      throw std::invalid_argument("Unsupported backend.");
    }
    return backend_it->second;
  } catch (const std::exception & e) {
    return std::nullopt;
  }
}

using DetectObjectsAction = isaac_manipulator_interfaces::action::DetectObjects;
using DopeAction = isaac_manipulator_interfaces::action::EstimatePoseDope;
using FoundationPoseAction = isaac_manipulator_interfaces::action::EstimatePoseFoundationPose;

using namespace std::chrono_literals;
}  // namespace

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

template<typename ActionType, typename BackendType>
std::optional<typename ActionType::Result::SharedPtr> ObjectInfoServer::Trigger(
  const typename ActionType::Goal & goal)
{
  auto action_client = action_client_manager_->GetRegisteredClient<BackendType, ActionType>();
  auto client = action_client->GetClient();
  if (!client->wait_for_action_server(std::chrono::seconds(10s))) {
    RCLCPP_ERROR(
      this->get_logger(), "Server (%s) not available after waiting %ld seconds. Aborting goal.",
      action_client->GetActionName().c_str(), std::chrono::seconds(10s).count());
    return std::nullopt;
  }
  auto goal_handle_future = client->async_send_goal(goal);
  auto status_goal_handle = goal_handle_future.wait_for(std::chrono::seconds(10s));
  if (status_goal_handle != std::future_status::ready) {
    RCLCPP_ERROR(
      this->get_logger(), "Goal Rejected for server(%s). Aborting goal.",
      action_client->GetActionName().c_str());
    return std::nullopt;
  }
  auto client_goal_handle = goal_handle_future.get();
  if (!client_goal_handle) {
    RCLCPP_ERROR(
      this->get_logger(), "Goal was rejected by server(%s)",
      action_client->GetActionName().c_str());
    return std::nullopt;
  }
  auto client_result_future = client->async_get_result(client_goal_handle);
  auto status_result = client_result_future.wait_for(std::chrono::seconds(10s));
  if (status_result != std::future_status::ready) {
    // Cancel the goal request
    client->async_cancel_goal(client_goal_handle);
    RCLCPP_ERROR(
      this->get_logger(), "Failed to get result from server(%s). Time out after waiting"
      " %ld seconds. Aborting goal.",
      action_client->GetActionName().c_str(), std::chrono::seconds(10s).count());
    return std::nullopt;
  }
  auto wrapped_result = client_result_future.get();
  if (wrapped_result.code == rclcpp_action::ResultCode::SUCCEEDED) {
    RCLCPP_INFO(
      this->get_logger(), "Got Results from server(%s)",
      action_client->GetActionName().c_str());
    return wrapped_result.result;
  } else {
    RCLCPP_ERROR(
      this->get_logger(), "Failed to get result from server(%s)",
      action_client->GetActionName().c_str());
    return std::nullopt;
  }
}

template<>
void ObjectInfoServer::Execute(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<GetObjectsAction>> goal_handle)
{
  RCLCPP_INFO(get_logger(), "Executing goal");
  std::vector<isaac_manipulator_interfaces::msg::ObjectInfo> objects;
  auto result = std::make_shared<typename GetObjectsAction::Result>();

  if (backend_manager_->ValidateBackendConfig()) {
    auto backend = backend_manager_->GetBackend<ObjectDetectionBackend>();

    if (backend == ObjectDetectionBackend::RT_DETR) {
      auto registered_client = action_client_manager_->GetRegisteredClient<ObjectDetectionBackend,
          DetectObjectsAction>();
      auto client_goal = registered_client->GetGoal();

      auto maybe_trigger_result = Trigger<DetectObjectsAction, ObjectDetectionBackend>(client_goal);

      if (maybe_trigger_result) {
        auto trigger_result = std::move(*maybe_trigger_result);
        auto detections = trigger_result->detections.detections;
        for (uint32_t i = 0; i < detections.size(); i++) {
          objects.push_back(isaac_manipulator_interfaces::msg::ObjectInfo());
          objects.back().object_id = i;
          objects.back().detection_2d = detections[i];
          // Cache object info
          objects_[i] = objects.back();
        }
        result->objects = objects;
        goal_handle->succeed(result);
        RCLCPP_INFO(get_logger(), "Execution succeeded");
        return;
      } else {
        goal_handle->abort(result);
        return;
      }
    } else if (backend == ObjectDetectionBackend::DOPE) {
      auto registered_client = action_client_manager_->GetRegisteredClient<ObjectDetectionBackend,
          DopeAction>();
      auto client_goal = registered_client->GetGoal();

      auto maybe_trigger_result = Trigger<DopeAction, ObjectDetectionBackend>(client_goal);

      if (maybe_trigger_result) {
        auto trigger_result = std::move(*maybe_trigger_result);
        auto detections_2d = trigger_result->detections.detections;
        auto detections_3d = trigger_result->poses.detections;
        for (uint32_t i = 0; i < detections_2d.size(); i++) {
          objects.push_back(isaac_manipulator_interfaces::msg::ObjectInfo());
          objects.back().object_id = i;
          objects.back().detection_2d = detections_2d[i];
          objects.back().detection_3d = detections_3d[i];
          objects_[i] = objects.back();
        }
        result->objects = objects;
        goal_handle->succeed(result);
        RCLCPP_INFO(get_logger(), "Execution succeeded");
        return;
      } else {
        goal_handle->abort(result);
        return;
      }
    }
  } else {
    goal_handle->abort(result);
    return;
  }
}

template<>
void ObjectInfoServer::Execute<GetObjectPoseAction>(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<GetObjectPoseAction>> goal_handle)
{
  RCLCPP_INFO(get_logger(), "Executing goal");
  auto result = std::make_shared<typename GetObjectPoseAction::Result>();
  if (backend_manager_->ValidateBackendConfig()) {
    // Check if GetObjects action has been called before GetObjectPose
    auto object_id = goal_handle->get_goal()->object_id;
    const auto object_it = objects_.find(object_id);
    if (object_it == std::end(objects_)) {
      RCLCPP_ERROR(
        get_logger(), "Object id (%d) not found. Make sure to call GetObjects action first",
        object_id);
      goal_handle->abort(result);
      return;
    }

    // Find pose in the ObjectInfo cache before triggering the pose estimation action
    auto object_info = object_it->second;
    auto zero_check = [](const double value) {
        return std::fabs(value) < 1e-6 ? true : false;
      };
    auto bbox_size = object_info.detection_3d.bbox.size;
    if (zero_check(bbox_size.x) && zero_check(bbox_size.y) && zero_check(bbox_size.z)) {
      RCLCPP_INFO(
        get_logger(), "Can't find detections_3d for object_id [%d]."
        " Calling pose estimation action.", object_id);
      // Trigger pose estimation action
      auto backend = backend_manager_->GetBackend<PoseEstimationBackend>();
      if (backend == PoseEstimationBackend::FOUNDATION_POSE) {
        auto registered_client = action_client_manager_->GetRegisteredClient<PoseEstimationBackend,
            FoundationPoseAction>();
        auto client_goal = registered_client->GetGoal();
        client_goal.roi = object_info.detection_2d;
        auto maybe_trigger_result =
          Trigger<FoundationPoseAction, PoseEstimationBackend>(client_goal);

        if (maybe_trigger_result) {
          auto trigger_result = std::move(*maybe_trigger_result);
          // Store the pose in the cache
          auto detection_3d = trigger_result->poses.detections[0];
          objects_[object_id].detection_3d = detection_3d;
          result->object_pose = detection_3d.results[0].pose.pose;
          goal_handle->succeed(result);
          return;
        } else {
          goal_handle->abort(result);
          return;
        }
      }
      return;
    } else {
      RCLCPP_INFO(
        get_logger(), "Found detections_3d for object_id [%d], Skipping pose estimation action.",
        object_id);
      auto detection_3d = object_info.detection_3d;
      result->object_pose = detection_3d.results[0].pose.pose;
      goal_handle->succeed(result);
      return;
    }

  } else {
    goal_handle->abort(result);
    return;
  }
}

template<>
rclcpp_action::GoalResponse ObjectInfoServer::HandleGoal<GetObjectsAction::Goal>(
  const rclcpp_action::GoalUUID & uuid, std::shared_ptr<const GetObjectsAction::Goal> goal)
{
  (void)goal;
  RCLCPP_INFO(get_logger(), "Received goal request for GetObjects");
  RCLCPP_INFO(get_logger(), "UUID: %s", rclcpp_action::to_string(uuid).c_str());

  if (backend_manager_->ValidateBackendConfig()) {
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  } else {
    RCLCPP_ERROR(this->get_logger(), "Invalid backend configuration. Rejecting goal.");
    return rclcpp_action::GoalResponse::REJECT;
  }
}

template<>
rclcpp_action::GoalResponse ObjectInfoServer::HandleGoal<GetObjectPoseAction::Goal>(
  const rclcpp_action::GoalUUID & uuid, std::shared_ptr<const GetObjectPoseAction::Goal> goal)
{
  RCLCPP_INFO(
    get_logger(), "Received goal request for GetObjectPose with object_id %d",
    goal->object_id);
  RCLCPP_INFO(get_logger(), "UUID: %s", rclcpp_action::to_string(uuid).c_str());

  if (backend_manager_->ValidateBackendConfig()) {
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  } else {
    RCLCPP_ERROR(this->get_logger(), "Invalid backend configuration. Rejecting goal.");
    return rclcpp_action::GoalResponse::REJECT;
  }
}

template<>
rclcpp_action::CancelResponse ObjectInfoServer::HandleCancel<GetObjectsAction>(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<GetObjectsAction>> goal_handle)
{
  RCLCPP_INFO(get_logger(), "Received request to cancel goal for GetObjects");
  (void)goal_handle;
  // Cancel internal goal
  auto backend = backend_manager_->GetBackend<ObjectDetectionBackend>();
  if (backend == ObjectDetectionBackend::RT_DETR) {
    auto client = action_client_manager_->GetRegisteredClient<ObjectDetectionBackend,
        DetectObjectsAction>()->GetClient();
    client->async_cancel_all_goals();
  }

  if (backend == ObjectDetectionBackend::DOPE) {
    auto client = action_client_manager_->GetRegisteredClient<ObjectDetectionBackend,
        DopeAction>()->GetClient();
    client->async_cancel_all_goals();
  }

  return rclcpp_action::CancelResponse::ACCEPT;
}

template<>
rclcpp_action::CancelResponse ObjectInfoServer::HandleCancel<GetObjectPoseAction>(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<GetObjectPoseAction>> goal_handle)
{
  RCLCPP_INFO(
    get_logger(),
    "Received request to cancel goal for GetObjectPose with object_id %d",
    goal_handle->get_goal()->object_id);
  (void)goal_handle;

  // Cancel internal goal
  auto backend = backend_manager_->GetBackend<PoseEstimationBackend>();
  if (backend == PoseEstimationBackend::FOUNDATION_POSE) {
    auto client = action_client_manager_->GetRegisteredClient<PoseEstimationBackend,
        FoundationPoseAction>()->GetClient();
    client->async_cancel_all_goals();
  }

  if (backend == PoseEstimationBackend::DOPE) {
    auto client = action_client_manager_->GetRegisteredClient<PoseEstimationBackend,
        DopeAction>()->GetClient();
    client->async_cancel_all_goals();
  }

  return rclcpp_action::CancelResponse::ACCEPT;
}

template<typename ActionType>
void ObjectInfoServer::HandleAccepted(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<ActionType>> goal_handle)
{
  // This needs to return quickly to avoid blocking the executor, so spin up a new thread
  std::thread{std::bind(
      &ObjectInfoServer::Execute<ActionType>, this, std::placeholders::_1),
    goal_handle}.detach();
}

void ObjectInfoServer::ClearObjects(
  const std::shared_ptr<isaac_manipulator_interfaces::srv::ClearObjects::Request> request,
  std::shared_ptr<isaac_manipulator_interfaces::srv::ClearObjects::Response> response)
{
  if (request->object_ids.empty()) {
    RCLCPP_INFO(get_logger(), "Clearing all objects in the cache.");
    response->count = objects_.size();
    objects_.clear();
    return;
  }

  auto count = 0;
  for (const auto id : request->object_ids) {
    const auto object_it = objects_.find(id);
    if (object_it != std::end(objects_)) {
      RCLCPP_INFO(get_logger(), "Clearing object with id [%d] in the cache.", id);
      objects_.erase(object_it);
      count++;
    } else {
      RCLCPP_WARN(get_logger(), "Object with id [%d] not found in the cache.", id);
    }
  }
  response->count = count;
}

ObjectInfoServer::ObjectInfoServer(const rclcpp::NodeOptions & options)
: Node("object_info_server", options),
  action_client_manager_(std::make_shared<ActionClientManager>()),
  pose_estimation_backend_(declare_parameter<std::string>(
      "pose_estimation_backend", "FOUNDATION_POSE")),
  object_detection_backend_(declare_parameter<std::string>(
      "object_detection_backend", "RT_DETR")),
  param_event_handler_(std::make_shared<rclcpp::ParameterEventHandler>(this)),
  get_objects_cb_group_(create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive)),
  get_objects_pose_cb_group_(create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive)),
  estimate_pose_fp_cb_group_(create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive)),
  estimate_pose_dope_cb_group_(create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive)),
  detect_objects_cb_group_(create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive)),
  clear_objects_cb_group_(create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive))
{
  clear_objects_ = create_service<isaac_manipulator_interfaces::srv::ClearObjects>(
    "/clear_objects",
    std::bind(
      &ObjectInfoServer::ClearObjects, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default,
    clear_objects_cb_group_);

  backend_manager_ = std::make_shared<BackendManager>();

  auto pose_estimation_backend = ValidateBackend(pose_estimation_backend_, POSE_ESTIMATION_BACKEND);
  if (pose_estimation_backend) {
    backend_manager_->AddBackend(*pose_estimation_backend);
    RegisterActionClients(*pose_estimation_backend);
  } else {
    RCLCPP_ERROR(
      this->get_logger(), "Invalid pose estimation backend [%s]",
      pose_estimation_backend_.c_str());
    throw std::invalid_argument("Invalid pose estimation backend.");
  }

  auto object_detection_backend = ValidateBackend(
    object_detection_backend_, OBJECT_DETECTION_BACKEND);
  if (object_detection_backend) {
    backend_manager_->AddBackend(*object_detection_backend);
    RegisterActionClients(*object_detection_backend);
  } else {
    RCLCPP_ERROR(
      this->get_logger(), "Invalid object detection backend [%s]",
      pose_estimation_backend_.c_str());
    throw std::invalid_argument("Invalid object detection backend.");
  }

  param_event_cb_ = param_event_handler_->add_parameter_event_callback(
    std::bind(&ObjectInfoServer::ParameterCallback, this, std::placeholders::_1));

  get_objects_server_ = rclcpp_action::create_server<GetObjectsAction>(
    this,
    "/get_objects",
    std::bind(
      &ObjectInfoServer::HandleGoal<GetObjectsAction::Goal>, this,
      std::placeholders::_1, std::placeholders::_2),
    std::bind(&ObjectInfoServer::HandleCancel<GetObjectsAction>, this, std::placeholders::_1),
    std::bind(&ObjectInfoServer::HandleAccepted<GetObjectsAction>, this, std::placeholders::_1),
    rcl_action_server_get_default_options(),
    get_objects_cb_group_);

  object_info_server_ = rclcpp_action::create_server<GetObjectPoseAction>(
    this,
    "/get_object_pose",
    std::bind(
      &ObjectInfoServer::HandleGoal<GetObjectPoseAction::Goal>, this,
      std::placeholders::_1, std::placeholders::_2),
    std::bind(
      &ObjectInfoServer::HandleCancel<GetObjectPoseAction>, this,
      std::placeholders::_1),
    std::bind(
      &ObjectInfoServer::HandleAccepted<GetObjectPoseAction>, this,
      std::placeholders::_1),
    rcl_action_server_get_default_options(),
    get_objects_pose_cb_group_);
}

void ObjectInfoServer::ParameterCallback(const rcl_interfaces::msg::ParameterEvent & event)
{
  for (const auto & changed_param : event.changed_parameters) {
    RCLCPP_INFO(
      get_logger(), "Changed parameter [%s = %s]", changed_param.name.c_str(),
      changed_param.value.string_value.c_str());
    const std::string param_value = changed_param.value.string_value;

    if (changed_param.name == "pose_estimation_backend") {
      auto backend = ValidateBackend(param_value, POSE_ESTIMATION_BACKEND);
      if (backend) {
        pose_estimation_backend_ = param_value;
        backend_manager_->AddBackend(*backend);
        RegisterActionClients(*backend);
        RCLCPP_INFO(
          get_logger(), "Changed pose estimation backend to [%s]",
          pose_estimation_backend_.c_str());
      } else {
        set_parameter(
          rclcpp::Parameter(changed_param.name, pose_estimation_backend_));
        RCLCPP_WARN(
          get_logger(), "Invalid backend [%s] for parameter [%s]. Reverted to prev[%s]",
          param_value.c_str(), changed_param.name.c_str(),
          pose_estimation_backend_.c_str());
      }
    } else if (changed_param.name == "object_detection_backend") {
      auto backend = ValidateBackend(param_value, OBJECT_DETECTION_BACKEND);
      if (backend) {
        object_detection_backend_ = param_value;
        backend_manager_->AddBackend(*backend);
        RegisterActionClients(*backend);
        RCLCPP_INFO(
          get_logger(), "Changed object detection backend to [%s]",
          object_detection_backend_.c_str());
      } else {
        set_parameter(
          rclcpp::Parameter(changed_param.name, object_detection_backend_));
        RCLCPP_WARN(
          get_logger(), "Invalid backend [%s] for parameter [%s]. Reverted to prev[%s]",
          param_value.c_str(), changed_param.name.c_str(),
          object_detection_backend_.c_str());
      }
    }

    if (backend_manager_->ValidateBackendConfig()) {
      RCLCPP_INFO(get_logger(), "Valid backend configuration");
    } else {
      RCLCPP_WARN(get_logger(), "Invalid backend configuration");
    }
  }
}

void ObjectInfoServer::RegisterActionClients(PoseEstimationBackend backend)
{
  if (backend == PoseEstimationBackend::FOUNDATION_POSE) {
    action_client_manager_->RegisterActionClient<PoseEstimationBackend>(
      std::make_shared<ActionClient<FoundationPoseAction>>(
        *this, "/estimate_pose_foundation_pose", estimate_pose_fp_cb_group_));
  } else if (backend == PoseEstimationBackend::DOPE) {
    action_client_manager_->RegisterActionClient<PoseEstimationBackend>(
      std::make_shared<ActionClient<DopeAction>>(
        *this, "/estimate_pose_dope", estimate_pose_dope_cb_group_));
  }
}

void ObjectInfoServer::RegisterActionClients(ObjectDetectionBackend backend)
{
  if (backend == ObjectDetectionBackend::RT_DETR) {
    action_client_manager_->RegisterActionClient<ObjectDetectionBackend>(
      std::make_shared<ActionClient<DetectObjectsAction>>(
        *this, "/detect_objects", detect_objects_cb_group_));
  } else if (backend == ObjectDetectionBackend::DOPE) {
    action_client_manager_->RegisterActionClient<ObjectDetectionBackend>(
      std::make_shared<ActionClient<DopeAction>>(
        *this, "/estimate_pose_dope", estimate_pose_dope_cb_group_));
  }
}

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac::manipulation::ObjectInfoServer)
