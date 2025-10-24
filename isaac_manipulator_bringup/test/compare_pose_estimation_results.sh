#!/bin/bash
#####################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
#####################################################################################
set -ex

if [ -z "$ISAAC_ROS_WS" ]; then
    echo "Error: ISAAC_ROS_WS environment variable is not set" >&2
    exit 1
fi

# Check if ENABLE_MANIPULATOR_TESTING is set to 'manual_on_robot'
if [ "$ENABLE_MANIPULATOR_TESTING" != "manual_on_robot" ]; then
    echo "Skipping test: ENABLE_MANIPULATOR_TESTING is not set to 'manual_on_robot'"
    echo "To run this test, set: export ENABLE_MANIPULATOR_TESTING=manual_on_robot"
    exit 0
fi

export OUTPUT_DIR=$ISAAC_ROS_WS/foundationstereo_pose_test
# Set it back to on_robot to run these tests
export ENABLE_MANIPULATOR_TESTING=on_robot

launch_test $ISAAC_ROS_WS/src/isaac_manipulator/isaac_manipulator_bringup/test/pick_and_place_servers_realsense_with_foundationstereo_test.py

export OUTPUT_DIR=$ISAAC_ROS_WS/ess_pose_test

launch_test $ISAAC_ROS_WS/src/isaac_manipulator/isaac_manipulator_bringup/test/pick_and_place_servers_realsense_with_ess_test.py

export OUTPUT_DIR=$ISAAC_ROS_WS/realsense_pose_test

launch_test $ISAAC_ROS_WS/src/isaac_manipulator/isaac_manipulator_bringup/test/pick_and_place_servers_realsense_pol_test.py

# Now that all 3 folders exist, then we run the evaluation.

mkdir -p $ISAAC_ROS_WS/pose_estimates

cp $ISAAC_ROS_WS/foundationstereo_pose_test/poses.json $ISAAC_ROS_WS/pose_estimates/fs_pose.json
cp $ISAAC_ROS_WS/ess_pose_test/poses.json $ISAAC_ROS_WS/pose_estimates/ess_pose.json
cp $ISAAC_ROS_WS/realsense_pose_test/poses.json $ISAAC_ROS_WS/pose_estimates/rs_pose.json

export POSE_FOLDER=$ISAAC_ROS_WS/pose_estimates
# Set it back to manual_on_robot for the next test
export ENABLE_MANIPULATOR_TESTING=manual_on_robot

launch_test $ISAAC_ROS_WS/src/isaac_manipulator/isaac_manipulator_bringup/test/pose_differences_depth_backend_test.py
