# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22.1)
project(isaac_manipulator_servers LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)

ament_auto_find_build_dependencies()

# Object Detection Server
ament_auto_add_library(
  object_detection_server SHARED
  src/object_detection_server.cpp
)

rclcpp_components_register_nodes(object_detection_server "nvidia::isaac::manipulation::ObjectDetectionServer")
set(node_plugins "${node_plugins}nvidia::isaac::manipulation::ObjectDetectionServer;$<TARGET_FILE:object_detection_server>\n")
set_target_properties(object_detection_server PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# Foundation Pose Server
ament_auto_add_library(
  foundation_pose_server SHARED
  src/foundation_pose_server.cpp
)

rclcpp_components_register_nodes(foundation_pose_server "nvidia::isaac::manipulation::FoundationPoseServer")
set(node_plugins "${node_plugins}nvidia::isaac::manipulation::FoundationPoseServer;$<TARGET_FILE:foundation_pose_server>\n")
set_target_properties(foundation_pose_server PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# Dope Server
ament_auto_add_library(
  dope_server SHARED
  src/dope_server.cpp
)

rclcpp_components_register_nodes(dope_server "nvidia::isaac::manipulation::DopeServer")
set(node_plugins "${node_plugins}nvidia::isaac::manipulation::DopeServer;$<TARGET_FILE:dope_server>\n")
set_target_properties(dope_server PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# Object Info Server
ament_auto_add_library(
  pick_and_place_server SHARED
  src/object_info_server.cpp
)

rclcpp_components_register_nodes(pick_and_place_server "nvidia::isaac::manipulation::ObjectInfoServer")
set(node_plugins "${node_plugins}nvidia::isaac::manipulation::ObjectInfoServer;$<TARGET_FILE:pick_and_place_server>\n")
set_target_properties(pick_and_place_server PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)

endif()


# Embed versioning information into installed files
ament_index_get_resource(ISAAC_ROS_COMMON_CMAKE_PATH isaac_ros_common_cmake_path isaac_ros_common)
include("${ISAAC_ROS_COMMON_CMAKE_PATH}/isaac_ros_common-version-info.cmake")
generate_version_info(${PROJECT_NAME})

ament_auto_package(INSTALL_TO_SHARE launch)
