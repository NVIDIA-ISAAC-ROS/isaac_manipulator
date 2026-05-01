#!/usr/bin/env bash

# 1. Source ROS environments so they're set in the shell
source /opt/ros/$ROS_DISTRO/setup.bash
source $ISAAC_ROS_WS/install/setup.bash
# This might be required below if nsys cannot find ros2 libraries
# For most cases it should not be needed.
# export LD_LIBRARY_PATH=<YOUR_LD_LIBRARY_CONTENTS>
# 2. Run nsys under sudo, forwarding all arguments passed to this script
exec sudo -E env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" nsys \
    profile --sample=system-wide \
            --run-as=admin \
            --gpuctxsw=true \
            --trace=osrt,nvtx,cuda \
            --gpu-metrics-devices=all \
            --delay 5 \
            --duration 100 \
            --stats=true \
            -o $ISAAC_ROS_WS/container_on_sim_system_wide.nsys \
    "$@"