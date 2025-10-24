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

#ifndef ISAAC_MANIPULATOR_SERVERS__IMPL__BACKEND_MANAGER_HPP_
#define ISAAC_MANIPULATOR_SERVERS__IMPL__BACKEND_MANAGER_HPP_

#include <string>
#include <unordered_map>
#include <variant>

#include "isaac_manipulator_servers/impl/backend_types.hpp"


namespace nvidia
{
namespace isaac
{
namespace manipulation
{

using backend_variants = std::variant<ObjectDetectionBackend, PoseEstimationBackend,
    SegmentationBackend>;

class BackendManager
{
public:
  BackendManager() = default;
  ~BackendManager() = default;

  bool ValidateBackendConfig() const;

  template<typename BackendType>
  void AddBackend(const BackendType backend);

  template<typename BackendType>
  BackendType GetBackend() const;

private:
  std::unordered_map<std::string, backend_variants> backend_map_;
};

template<typename BackendType>
void BackendManager::AddBackend(const BackendType backend)
{
  std::string key = typeid(BackendType).name();
  backend_map_[key] = backend_variants(backend);
}

template<typename BackendType>
BackendType BackendManager::GetBackend() const
{
  std::string key = typeid(BackendType).name();
  return std::get<BackendType>(backend_map_.at(key));
}

bool BackendManager::ValidateBackendConfig() const
{
  try {
    return (GetBackend<PoseEstimationBackend>() == PoseEstimationBackend::FOUNDATION_POSE &&
           GetBackend<ObjectDetectionBackend>() == ObjectDetectionBackend::RT_DETR) ||
           (GetBackend<PoseEstimationBackend>() == PoseEstimationBackend::FOUNDATION_POSE &&
           GetBackend<ObjectDetectionBackend>() == ObjectDetectionBackend::GROUNDING_DINO) ||
           (GetBackend<PoseEstimationBackend>() == PoseEstimationBackend::DOPE &&
           GetBackend<ObjectDetectionBackend>() == ObjectDetectionBackend::DOPE) ||
           (GetBackend<PoseEstimationBackend>() == PoseEstimationBackend::FOUNDATION_POSE &&
           GetBackend<SegmentationBackend>() == SegmentationBackend::SEGMENT_ANYTHING &&
           GetBackend<ObjectDetectionBackend>() == ObjectDetectionBackend::RT_DETR) ||
           (GetBackend<PoseEstimationBackend>() == PoseEstimationBackend::FOUNDATION_POSE &&
           GetBackend<SegmentationBackend>() == SegmentationBackend::SEGMENT_ANYTHING &&
           GetBackend<ObjectDetectionBackend>() == ObjectDetectionBackend::GROUNDING_DINO) ||
           (GetBackend<PoseEstimationBackend>() == PoseEstimationBackend::FOUNDATION_POSE &&
           GetBackend<SegmentationBackend>() == SegmentationBackend::SEGMENT_ANYTHING &&
           GetBackend<ObjectDetectionBackend>() == ObjectDetectionBackend::SEGMENT_ANYTHING) ||
           (GetBackend<PoseEstimationBackend>() == PoseEstimationBackend::FOUNDATION_POSE &&
           GetBackend<SegmentationBackend>() == SegmentationBackend::SEGMENT_ANYTHING2 &&
           GetBackend<ObjectDetectionBackend>() == ObjectDetectionBackend::RT_DETR) ||
           (GetBackend<PoseEstimationBackend>() == PoseEstimationBackend::FOUNDATION_POSE &&
           GetBackend<SegmentationBackend>() == SegmentationBackend::SEGMENT_ANYTHING2 &&
           GetBackend<ObjectDetectionBackend>() == ObjectDetectionBackend::GROUNDING_DINO) ||
           (GetBackend<PoseEstimationBackend>() == PoseEstimationBackend::FOUNDATION_POSE &&
           GetBackend<SegmentationBackend>() == SegmentationBackend::SEGMENT_ANYTHING2 &&
           GetBackend<ObjectDetectionBackend>() == ObjectDetectionBackend::SEGMENT_ANYTHING2);
  } catch (const std::out_of_range &) {
    // If a backend type is not found in the map, return false
    return false;
  } catch (const std::bad_variant_access &) {
    // If a backend type is not in the variant, return false
    return false;
  }
}

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

#endif  // ISAAC_MANIPULATOR_SERVERS__IMPL__BACKEND_MANAGER_HPP_
