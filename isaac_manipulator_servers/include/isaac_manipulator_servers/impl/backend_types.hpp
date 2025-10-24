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

#ifndef ISAAC_MANIPULATOR_SERVERS__IMPL__BACKEND_TYPES_HPP_
#define ISAAC_MANIPULATOR_SERVERS__IMPL__BACKEND_TYPES_HPP_

namespace nvidia
{
namespace isaac
{
namespace manipulation
{

#include <iostream>
#include <map>
#include <string_view>

enum class PoseEstimationBackend
{
  DOPE,
  FOUNDATION_POSE
};

enum class ObjectDetectionBackend
{
  DOPE,
  GROUNDING_DINO,
  RT_DETR,
  SEGMENT_ANYTHING,
  SEGMENT_ANYTHING2
};

enum class SegmentationBackend
{
  SEGMENT_ANYTHING,
  SEGMENT_ANYTHING2,
  NONE
};

}  // namespace manipulation
}  // namespace isaac
}  // namespace nvidia

#endif  // ISAAC_MANIPULATOR_SERVERS__IMPL__BACKEND_TYPES_HPP_
