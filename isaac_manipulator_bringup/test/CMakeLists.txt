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

find_package(launch_testing_ament_cmake REQUIRED)

# Get the path to a dummy bag for use in the tests.
get_dummy_bag_path(DUMMY_BAG_PATH)

# Get the ISAAC_ROS_WS path.
set(ISAAC_ROS_WS "$ENV{ISAAC_ROS_WS}")

# NOTE: We test the pose to pose example with realsense and hawk.
#       Additionally, we change the num_cameras, setup, no_robot_mode arguments
#       to increase test coverage.
# TODO: Add tests for the object following example.
#       Currently, we are missing an install script for the required models
#       (similar to install_ess_models_script).
add_graph_startup_test(realsense_pose_to_pose_dry_run
    launch/cumotion_nvblox_pose_to_pose.launch.py
    TIMEOUT 20
    ARGS
"camera_type:=realsense num_cameras:=2 setup:=zurich_test_bench \
run_rviz:=False run_foxglove:=False log_level:=info disable_esdf_visualizer:=True no_robot_mode:=True enable_test_keepalive:=True disable_cameras:=True"
)

add_graph_startup_test(hawk_pose_to_pose_dry_run
    launch/cumotion_nvblox_pose_to_pose.launch.py
    TIMEOUT 20
    ARGS
"camera_type:=hawk no_robot_mode:=True \
ess_engine_file_path:=${ISAAC_ROS_WS}/isaac_ros_assets/models/dnn_stereo_disparity/dnn_stereo_disparity_v4.1.0_onnx/ess.engine \
setup:=zurich_test_bench run_rviz:=False run_foxglove:=False log_level:=info disable_esdf_visualizer:=True enable_test_keepalive:=True disable_cameras:=True"
)

# Add installation of ESS models as a test fixture.
get_isaac_ros_asset_install_script_path(install_ess_models_script install_ess_models isaac_ros_ess_models_install)
ament_add_test(setup_ess_models
    COMMAND "${install_ess_models_script}"
    ENV "ISAAC_ROS_SKIP_ASSET_INSTALL="
    TIMEOUT 200
    GENERATE_RESULT_FOR_RETURN_CODE_ZERO
)

# Register the setup ess models script as a pre-test requirement of the hawk dry run.
set_tests_properties(setup_ess_models PROPERTIES FIXTURES_SETUP pre_test)
set_tests_properties(hawk_pose_to_pose_dry_run PROPERTIES FIXTURES_REQUIRED pre_test)
