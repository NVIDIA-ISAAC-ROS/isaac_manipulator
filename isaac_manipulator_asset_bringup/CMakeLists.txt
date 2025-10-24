
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(isaac_manipulator_asset_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(rclcpp_action REQUIRED)
ament_auto_find_build_dependencies()

install(
  PROGRAMS scripts/setup_perception_models.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()
  add_subdirectory(test)

  # Custom target to run the Python script that downloads all test assets
  add_custom_target(
    setup_perception_models ALL
    COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/scripts/setup_perception_models.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running script to setup assets for Isaac Manipulator"
  )
endif()

ament_auto_package(INSTALL_TO_SHARE scripts)