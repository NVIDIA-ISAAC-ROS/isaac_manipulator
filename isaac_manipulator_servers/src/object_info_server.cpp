// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <unistd.h>

#include <chrono>
#include <cmath>
#include <vector>

#include "isaac_manipulator_servers/object_info_server.hpp"
#include "isaac_manipulator_servers/impl/action_clients.hpp"
#include "isaac_manipulator_servers/impl/backend_types.hpp"

namespace
{
using PoseEstimationBackend = nvidia::isaac::manipulation::PoseEstimationBackend;
using ObjectDetectionBackend = nvidia::isaac::manipulation::ObjectDetectionBackend;
using SegmentationBackend = nvidia::isaac::manipulation::SegmentationBackend;

// User string to pose estimation backend mode
const std::unordered_map<std::string, PoseEstimationBackend> POSE_ESTIMATION_BACKEND({
    {"DOPE", PoseEstimationBackend::DOPE},
    {"FOUNDATION_POSE", PoseEstimationBackend::FOUNDATION_POSE}
  });

// User string to object detection backend mode
const std::unordered_map<std::string, ObjectDetectionBackend> OBJECT_DETECTION_BACKEND({
    {"DOPE", ObjectDetectionBackend::DOPE},
    {"GROUNDING_DINO", ObjectDetectionBackend::GROUNDING_DINO},
    {"RT_DETR", ObjectDetectionBackend::RT_DETR},
    {"SEGMENT_ANYTHING", ObjectDetectionBackend::SEGMENT_ANYTHING},
    {"SEGMENT_ANYTHING2", ObjectDetectionBackend::SEGMENT_ANYTHING2}
  });

// User string to segmentation backend mode
const std::unordered_map<std::string, SegmentationBackend> SEGMENTATION_BACKEND({
    {"SEGMENT_ANYTHING", SegmentationBackend::SEGMENT_ANYTHING},
    {"SEGMENT_ANYTHING2", SegmentationBackend::SEGMENT_ANYTHING2},
    {"NONE", SegmentationBackend::NONE}
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

using AddSegmentationMaskAction = isaac_manipulator_interfaces::action::AddSegmentationMask;
using DetectObjectsAction = isaac_manipulator_interfaces::action::DetectObjects;
using DopeAction = isaac_manipulator_interfaces::action::EstimatePoseDope;
using FoundationPoseAction = isaac_manipulator_interfaces::action::EstimatePoseFoundationPose;
using SegmentAnythingAction = isaac_manipulator_interfaces::action::SegmentAnything;

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
  // Lock the objects_mutex_ to prevent race condition for multiple action calls
  // At the same time.
  std::lock_guard<std::mutex> lock(objects_mutex_);

  RCLCPP_INFO(get_logger(), "Executing goal");
  std::vector<isaac_manipulator_interfaces::msg::ObjectInfo> objects;
  auto result = std::make_shared<typename GetObjectsAction::Result>();

  if (backend_manager_->ValidateBackendConfig()) {
    auto backend = backend_manager_->GetBackend<ObjectDetectionBackend>();

    if (backend == ObjectDetectionBackend::GROUNDING_DINO ||
      backend == ObjectDetectionBackend::RT_DETR)
    {
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
          objects.back().has_segmentation_mask = false;

          // Cache object info with new ID
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
    } else if (backend == ObjectDetectionBackend::SEGMENT_ANYTHING ||  // NOLINT
      backend == ObjectDetectionBackend::SEGMENT_ANYTHING2)
    {
      // Get the goal parameters
      const auto goal = goal_handle->get_goal();

      // Check if use_initial_hint is true
      if (!goal->use_initial_hint) {
        // If not using initial hint, return existing objects
        if (objects_.empty()) {
          RCLCPP_ERROR(
            get_logger(),
            "No objects in cache and no initial hint provided");
          goal_handle->abort(result);
          return;
        }

        // Convert cached objects to result
        for (const auto & [id, obj] : objects_) {
          objects.push_back(obj);
        }

        // Now remove segmentation mask from return variable before sending it to the user
        for (auto & obj : objects) {
          obj.segmentation_mask.data.clear();  // Clearing just for the result not in main cache
          obj.has_segmentation_mask = true;
        }

        result->objects = objects;
        goal_handle->succeed(result);
        RCLCPP_INFO(get_logger(), "Returned %zu existing objects", objects.size());
        return;
      }

      // Get the initial hint point
      vision_msgs::msg::Point2D initial_hint;
      initial_hint.x = goal->initial_hint.x;
      initial_hint.y = goal->initial_hint.y;

      // Create and send DetectObject goal
      auto registered_client = action_client_manager_->GetRegisteredClient<ObjectDetectionBackend,
          SegmentAnythingAction>();
      auto client_goal = registered_client->GetGoal();

      client_goal.initial_hint_point = initial_hint;
      client_goal.use_point_hint = true;

      auto maybe_trigger_result =
        Trigger<SegmentAnythingAction, SegmentationBackend>(client_goal);

      if (maybe_trigger_result) {
        auto trigger_result = std::move(*maybe_trigger_result);

        // Create object info and cache it
        objects.push_back(isaac_manipulator_interfaces::msg::ObjectInfo());
        objects.back().object_id = objects_.size();  // Use size as next ID
        objects.back().detection_2d = trigger_result->detection;
        objects.back().segmentation_mask = trigger_result->segmentation_mask;
        objects.back().has_segmentation_mask = true;

        // Cache the object
        objects_[objects.back().object_id] = objects.back();

        result->objects = objects;

        // Now remove segmentation mask from return variable before sending it to the user
        for (auto & obj : result->objects) {
          obj.segmentation_mask.data.clear();  // Clearing just for the result not in main cache
          obj.has_segmentation_mask = true;
        }

        goal_handle->succeed(result);
        RCLCPP_INFO(get_logger(), "Successfully detected object with initial hint using SAM");
        return;
      } else {
        RCLCPP_ERROR(get_logger(), "Failed to detect object with initial hint using SAM");
        goal_handle->abort(result);
        return;
      }
    }
  } else {
    RCLCPP_ERROR(this->get_logger(), "Invalid backend configuration. Rejecting goal.");
    goal_handle->abort(result);
    return;
  }
}

template<>
void ObjectInfoServer::Execute<GetObjectPoseAction>(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<GetObjectPoseAction>> goal_handle)
{
  std::lock_guard<std::mutex> lock(objects_mutex_);
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
      auto segmentation_backend = backend_manager_->GetBackend<SegmentationBackend>();
      if (backend == PoseEstimationBackend::FOUNDATION_POSE) {
        auto registered_client = action_client_manager_->GetRegisteredClient<PoseEstimationBackend,
            FoundationPoseAction>();
        auto client_goal = registered_client->GetGoal();

        if (segmentation_backend != SegmentationBackend::NONE) {
          client_goal.use_segmentation_mask = true;
          client_goal.segmentation_mask = object_info.segmentation_mask;

          if (object_info.segmentation_mask.data.empty()) {
            RCLCPP_ERROR(
              get_logger(), "Segmentation mask is empty for object_id [%d]", object_id);
            goal_handle->abort(result);
            return;
          }
        } else {
          client_goal.roi = object_info.detection_2d;
        }

        // Set mesh resource before triggering the pose estimation action
        if (!object_info.mesh_file_path.empty()) {
          client_goal.mesh_file_path = object_info.mesh_file_path;
        }

        // Set name of the object before triggering the pose estimation action
        if (!object_info.name.empty()) {
          client_goal.object_frame_name = object_info.name;
        }

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
      } else {
        RCLCPP_ERROR(
          get_logger(), "Invalid pose estimation backend [%s]", pose_estimation_backend_.c_str());
        throw std::invalid_argument("Invalid pose estimation backend.");
      }
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
    RCLCPP_ERROR(this->get_logger(), "Invalid backend configuration. Rejecting goal.");
    goal_handle->abort(result);
    return;
  }
}

template<>
void ObjectInfoServer::Execute<AddSegmentationMaskAction>(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<AddSegmentationMaskAction>> goal_handle)
{
  // Add this line to prevent race conditions
  std::lock_guard<std::mutex> lock(objects_mutex_);

  RCLCPP_INFO(get_logger(), "Executing goal for AddSegmentationMask");
  auto result = std::make_shared<typename AddSegmentationMaskAction::Result>();

  // Validate backend configuration
  if (!backend_manager_->ValidateBackendConfig()) {
    RCLCPP_ERROR(get_logger(), "Backend not configured correctly");
    goal_handle->abort(result);
    return;
  }

  // Check if object exists in cache
  auto object_id = goal_handle->get_goal()->object_id;
  RCLCPP_INFO(get_logger(), "Object id for AddSegmentationMask: %d", object_id);
  const auto object_it = objects_.find(object_id);
  if (object_it == std::end(objects_)) {
    RCLCPP_ERROR(
      get_logger(), "Object id (%d) not found. Make sure to call GetObjects action first",
      object_id);
    goal_handle->abort(result);
    return;
  }

  // Get the object info
  auto object_info = object_it->second;

  // If segmentation mask already exists, return it
  if (!object_info.segmentation_mask.data.empty()) {
    // Need to perform a copy here, not a move.
    // Potentially expensive CPU to CPU copy over here so just send the flag
    result->has_segmentation_mask = object_info.has_segmentation_mask;
    goal_handle->succeed(result);
    return;
  }

  // Check if we have detection_2d to use as initial hint
  vision_msgs::msg::BoundingBox2D initial_hint = object_info.detection_2d.bbox;
  if (initial_hint.size_x == 0 || initial_hint.size_y == 0 ||
    (initial_hint.center.position.x == 0 && initial_hint.center.position.y == 0))
  {
    RCLCPP_ERROR(
      get_logger(), "Invalid detection_2d found for object id (%d).",
      object_id);
    goal_handle->abort(result);
    return;
  }

  auto segmentation_backend = backend_manager_->GetBackend<SegmentationBackend>();
  if (segmentation_backend == SegmentationBackend::SEGMENT_ANYTHING ||
    segmentation_backend == SegmentationBackend::SEGMENT_ANYTHING2)
  {
    // Create and send SegmentAnything goal
    auto registered_client = action_client_manager_->GetRegisteredClient<SegmentationBackend,
        SegmentAnythingAction>();
    auto client_goal = registered_client->GetGoal();
    client_goal.initial_hint_bbox = initial_hint;
    client_goal.use_point_hint = false;

    RCLCPP_INFO(
      get_logger(), "Calling SegmentAnything with initial hint (x: %f, y: %f, w: %f, h: %f)",
      initial_hint.center.position.x, initial_hint.center.position.y,
      initial_hint.size_x, initial_hint.size_y);

    auto maybe_trigger_result = Trigger<SegmentAnythingAction, SegmentationBackend>(client_goal);

    if (maybe_trigger_result) {
      auto trigger_result = std::move(*maybe_trigger_result);

      // Then move into cache
      objects_[object_id].segmentation_mask = std::move(trigger_result->segmentation_mask);
      objects_[object_id].has_segmentation_mask = true;
      // Now copy over this big image to send the result to the user,
      // This is a very expensive operation so we don't do it, instead just send the flag
      result->has_segmentation_mask = true;

      goal_handle->succeed(result);
      RCLCPP_INFO(
        get_logger(), "Successfully generated segmentation mask for object id (%d)", object_id);
    } else {
      RCLCPP_ERROR(
        get_logger(), "Failed to generate segmentation mask for object id (%d)", object_id);
      goal_handle->abort(result);
    }

  } else {
    RCLCPP_ERROR(
      get_logger(), "Invalid segmentation backend [%s]", segmentation_backend_.c_str());
    throw std::invalid_argument("Invalid segmentation backend.");
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
rclcpp_action::GoalResponse ObjectInfoServer::HandleGoal<AddSegmentationMaskAction::Goal>(
  const rclcpp_action::GoalUUID & uuid, std::shared_ptr<const AddSegmentationMaskAction::Goal> goal)
{
  RCLCPP_INFO(
    get_logger(), "Received goal request for AddSegmentationMask with object_id %d",
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
  if (backend == ObjectDetectionBackend::GROUNDING_DINO ||
    backend == ObjectDetectionBackend::RT_DETR)
  {
    auto client = action_client_manager_->GetRegisteredClient<ObjectDetectionBackend,
        DetectObjectsAction>()->GetClient();
    client->async_cancel_all_goals();
  }

  if (backend == ObjectDetectionBackend::DOPE) {
    auto client = action_client_manager_->GetRegisteredClient<ObjectDetectionBackend,
        DopeAction>()->GetClient();
    client->async_cancel_all_goals();
  }

  if (backend == ObjectDetectionBackend::SEGMENT_ANYTHING) {
    auto client = action_client_manager_->GetRegisteredClient<ObjectDetectionBackend,
        SegmentAnythingAction>()->GetClient();
    client->async_cancel_all_goals();
  }

  if (backend == ObjectDetectionBackend::SEGMENT_ANYTHING2) {
    auto client = action_client_manager_->GetRegisteredClient<ObjectDetectionBackend,
        SegmentAnythingAction>()->GetClient();
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

template<>
rclcpp_action::CancelResponse ObjectInfoServer::HandleCancel<AddSegmentationMaskAction>(
  const std::shared_ptr<rclcpp_action::ServerGoalHandle<AddSegmentationMaskAction>> goal_handle)
{
  RCLCPP_INFO(
    get_logger(),
    "Received request to cancel goal for AddSegmentationMask with object_id %d",
    goal_handle->get_goal()->object_id);
  (void)goal_handle;

  // Cancel internal goal
  auto backend = backend_manager_->GetBackend<SegmentationBackend>();
  if (backend == SegmentationBackend::SEGMENT_ANYTHING ||
    backend == SegmentationBackend::SEGMENT_ANYTHING2)
  {
    auto client = action_client_manager_->GetRegisteredClient<SegmentationBackend,
        SegmentAnythingAction>()->GetClient();
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
  const std::shared_ptr<ClearObjectsSrv::Request> request,
  std::shared_ptr<ClearObjectsSrv::Response> response)
{
  std::lock_guard<std::mutex> lock(objects_mutex_);
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

void ObjectInfoServer::AddMeshToObject(
  const std::shared_ptr<AddMeshToObjectSrv::Request> request,
  std::shared_ptr<AddMeshToObjectSrv::Response> response)
{
  std::lock_guard<std::mutex> lock(objects_mutex_);
  response->success = true;
  response->failed_ids.clear();

  // Check if input arrays have same size
  if (request->object_ids.size() != request->mesh_file_paths.size()) {
    response->success = false;
    response->message = "Mismatched array sizes between object_ids and mesh_file_paths";
    return;
  }

  // Process each object ID and mesh file path pair
  for (size_t i = 0; i < request->object_ids.size(); ++i) {
    const auto object_id = request->object_ids[i];
    const auto & mesh_file_path = request->mesh_file_paths[i];

    // Check if object exists in cache
    auto object_it = objects_.find(object_id);
    if (object_it == std::end(objects_)) {
      RCLCPP_WARN(
        get_logger(), "Object with id [%d] not found in the cache.", object_id);
      response->failed_ids.push_back(object_id);
      response->success = false;
      continue;
    }

    // Check if mesh file exists
    if (access(mesh_file_path.c_str(), F_OK | R_OK) == -1) {
      RCLCPP_WARN(
        get_logger(), "Mesh file [%s] does not exist for object id [%d]",
        mesh_file_path.c_str(), object_id);
      response->failed_ids.push_back(object_id);
      response->success = false;
      continue;
    }

    // Update mesh file path
    object_it->second.mesh_file_path = mesh_file_path;
    RCLCPP_INFO(
      get_logger(), "Successfully added mesh file [%s] to object id [%d]",
      mesh_file_path.c_str(), object_id);
  }

  // Set appropriate message based on results
  if (response->failed_ids.empty()) {
    response->message = "Successfully added all mesh files";
  } else {
    response->message = "Failed to add mesh files for some objects. Check failed_ids";
  }
}

void ObjectInfoServer::AssignNameToObject(
  const std::shared_ptr<AssignNameToObjectSrv::Request> request,
  std::shared_ptr<AssignNameToObjectSrv::Response> response)
{
  std::lock_guard<std::mutex> lock(objects_mutex_);
  const auto object_id = request->object_id;
  const auto & name = request->name;

  // Check if object exists in cache
  auto object_it = objects_.find(object_id);
  if (object_it == std::end(objects_)) {
    RCLCPP_WARN(get_logger(), "Object with id [%d] not found in the cache.", object_id);
    response->result = false;
    return;
  }

  // Update object name
  object_it->second.name = name;
  response->result = true;
}

ObjectInfoServer::ObjectInfoServer(const rclcpp::NodeOptions & options)
: Node("object_info_server", options),
  action_client_manager_(std::make_shared<ActionClientManager>()),
  pose_estimation_backend_(declare_parameter<std::string>(
      "pose_estimation_backend", "FOUNDATION_POSE")),
  object_detection_backend_(declare_parameter<std::string>(
      "object_detection_backend", "RT_DETR")),
  segmentation_backend_(declare_parameter<std::string>(
      "segmentation_backend", "SEGMENT_ANYTHING")),
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
  segmentation_cb_group_(create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive)),
  clear_objects_cb_group_(create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive)),
  add_mesh_objects_cb_group_(create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive)),
  assign_name_to_object_cb_group_(create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive))
{
  clear_objects_ = create_service<ClearObjectsSrv>(
    "clear_objects",
    std::bind(
      &ObjectInfoServer::ClearObjects, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(),
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
      object_detection_backend_.c_str());
    throw std::invalid_argument("Invalid object detection backend.");
  }

  auto segmentation_backend = ValidateBackend(
    segmentation_backend_, SEGMENTATION_BACKEND);
  if (segmentation_backend) {
    backend_manager_->AddBackend(*segmentation_backend);
    RegisterActionClients(*segmentation_backend);
  } else {
    RCLCPP_ERROR(
      this->get_logger(), "Invalid segmentation backend [%s]",
      segmentation_backend_.c_str());
    throw std::invalid_argument("Invalid segmentation backend.");
  }

  param_event_cb_ = param_event_handler_->add_parameter_event_callback(
    std::bind(&ObjectInfoServer::ParameterCallback, this, std::placeholders::_1));

  get_objects_server_ = rclcpp_action::create_server<GetObjectsAction>(
    this,
    "get_objects",
    std::bind(
      &ObjectInfoServer::HandleGoal<GetObjectsAction::Goal>, this,
      std::placeholders::_1, std::placeholders::_2),
    std::bind(&ObjectInfoServer::HandleCancel<GetObjectsAction>, this, std::placeholders::_1),
    std::bind(&ObjectInfoServer::HandleAccepted<GetObjectsAction>, this, std::placeholders::_1),
    rcl_action_server_get_default_options(), get_objects_cb_group_);

  segmentation_server_ = rclcpp_action::create_server<AddSegmentationMaskAction>(
    this,
    "add_segmentation_mask",
    std::bind(
      &ObjectInfoServer::HandleGoal<AddSegmentationMaskAction::Goal>, this,
      std::placeholders::_1, std::placeholders::_2),
    std::bind(
      &ObjectInfoServer::HandleCancel<AddSegmentationMaskAction>, this, std::placeholders::_1),
    std::bind(
      &ObjectInfoServer::HandleAccepted<AddSegmentationMaskAction>, this, std::placeholders::_1),
    rcl_action_server_get_default_options(), segmentation_cb_group_);

  object_info_server_ = rclcpp_action::create_server<GetObjectPoseAction>(
    this,
    "get_object_pose",
    std::bind(
      &ObjectInfoServer::HandleGoal<GetObjectPoseAction::Goal>, this,
      std::placeholders::_1, std::placeholders::_2),
    std::bind(
      &ObjectInfoServer::HandleCancel<GetObjectPoseAction>, this,
      std::placeholders::_1),
    std::bind(
      &ObjectInfoServer::HandleAccepted<GetObjectPoseAction>, this,
      std::placeholders::_1),
    rcl_action_server_get_default_options(), get_objects_pose_cb_group_);

  add_mesh_objects_ = create_service<AddMeshToObjectSrv>(
    "add_mesh_to_object",
    std::bind(
      &ObjectInfoServer::AddMeshToObject, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(), add_mesh_objects_cb_group_);

  assign_name_to_object_ = create_service<AssignNameToObjectSrv>(
    "assign_name_to_object",
    std::bind(
      &ObjectInfoServer::AssignNameToObject, this, std::placeholders::_1, std::placeholders::_2),
    rclcpp::ServicesQoS(), assign_name_to_object_cb_group_);
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
    } else if (changed_param.name == "segmentation_backend") {
      auto backend = ValidateBackend(param_value, SEGMENTATION_BACKEND);
      if (backend) {
        segmentation_backend_ = param_value;
        backend_manager_->AddBackend(*backend);
        RegisterActionClients(*backend);
        RCLCPP_INFO(
          get_logger(), "Changed segmentation backend to [%s]",
          segmentation_backend_.c_str());
      } else {
        set_parameter(rclcpp::Parameter(changed_param.name, segmentation_backend_));
        RCLCPP_WARN(
          get_logger(), "Invalid backend [%s] for parameter [%s]. Reverted to prev[%s]",
          param_value.c_str(), changed_param.name.c_str(),
          segmentation_backend_.c_str());
      }
    }

    if (backend_manager_->ValidateBackendConfig()) {
      RCLCPP_INFO(get_logger(), "Valid backend configuration");
    } else {
      RCLCPP_ERROR(get_logger(), "Invalid backend configuration");
    }
  }
}

void ObjectInfoServer::RegisterActionClients(PoseEstimationBackend backend)
{
  if (backend == PoseEstimationBackend::FOUNDATION_POSE) {
    action_client_manager_->RegisterActionClient<PoseEstimationBackend>(
      std::make_shared<ActionClient<FoundationPoseAction>>(
        *this, "estimate_pose_foundation_pose", estimate_pose_fp_cb_group_));
  } else if (backend == PoseEstimationBackend::DOPE) {
    action_client_manager_->RegisterActionClient<PoseEstimationBackend>(
      std::make_shared<ActionClient<DopeAction>>(
        *this, "estimate_pose_dope", estimate_pose_dope_cb_group_));
  }
}

void ObjectInfoServer::RegisterActionClients(ObjectDetectionBackend backend)
{
  if (backend == ObjectDetectionBackend::GROUNDING_DINO ||
    backend == ObjectDetectionBackend::RT_DETR)
  {
    action_client_manager_->RegisterActionClient<ObjectDetectionBackend>(
      std::make_shared<ActionClient<DetectObjectsAction>>(
        *this, "detect_objects", detect_objects_cb_group_));
  } else if (backend == ObjectDetectionBackend::DOPE) {
    action_client_manager_->RegisterActionClient<ObjectDetectionBackend>(
      std::make_shared<ActionClient<DopeAction>>(
        *this, "estimate_pose_dope", estimate_pose_dope_cb_group_));
  } else if (backend == ObjectDetectionBackend::SEGMENT_ANYTHING ||  // NOLINT
    backend == ObjectDetectionBackend::SEGMENT_ANYTHING2)
  {
    action_client_manager_->RegisterActionClient<ObjectDetectionBackend>(
      std::make_shared<ActionClient<SegmentAnythingAction>>(
        *this, "segment_anything", segmentation_cb_group_));
  }
}

void ObjectInfoServer::RegisterActionClients(SegmentationBackend backend)
{
  if (backend == SegmentationBackend::SEGMENT_ANYTHING ||
    backend == SegmentationBackend::SEGMENT_ANYTHING2)
  {
    action_client_manager_->RegisterActionClient<SegmentationBackend>(
      std::make_shared<ActionClient<SegmentAnythingAction>>(
        *this, "segment_anything", segmentation_cb_group_));
  }
}

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

// Register as a component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac::manipulation::ObjectInfoServer)
