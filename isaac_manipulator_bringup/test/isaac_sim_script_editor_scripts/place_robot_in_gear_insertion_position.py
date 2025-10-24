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
"""
Script to place the robot in the gear insertion position in Isaac Sim.

First start Isaac Sim, run this action server in a new terminal:

1. ros2 run isaac_manipulator_isaac_sim_utils isaac_sim_gripper_driver.py
2. ros2 action send_goal \
        /robotiq_gripper_controller/gripper_cmd \
        control_msgs/action/GripperCommand "command: {position: 0.0, max_effort: 100.0}"
3. Paste this script in the Isaac Sim Script Editor.
4. Run the script.
5. Run it again so that gear and robot are in the correct position.
6. The close the gripper via this call.
    ros2 action send_goal \
        /robotiq_gripper_controller/gripper_cmd \
        control_msgs/action/GripperCommand "command: {position: 0.50, max_effort: 100.0}"
7. The run this test. The gear should get inserted in the peg.
8. It will wait for user input to start the insertion process. This is the cmd below:
   ros2 topic pub /input_points_debug geometry_msgs/msg/Point "{x: 0.0, y: 0.0, z: 0.0}"
9. ros2 topic pub /stop_joint_commands std_msgs/msg/Bool "data: true"
This script will place the robot in the gear insertion position.

"""

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, SingleXFormPrim

import numpy as np
from scipy.spatial.transform import Rotation as R


def set_gear_stand_at_desired_wrt_robot(
    desired_pos_G_in_RB:   list[float],    # [x, y, z] of gear relative to RB‐frame
    desired_quat_G_in_RB:  list[float],    # [w, x, y, z] of gear orientation in RB‐frame
):
    """
    Places /World/gear_assembly_frame so that its pose in R‐frame matches.

    It matches (desired_pos_G_in_robot_base, desired_quat_G_in_robot_base), given that
    R rotated by 180° about Z (no translation). It correlates to `base` in TF frame.
    """
    # 1) Read robot’s world pose (i.e. robot→world):
    robot_prim = SingleXFormPrim('/World/ur10e_robotiq2f_140_ROS')
    robot_pos_W, robot_quat_W = robot_prim.get_world_pose()
    # robot_quat_W is [w, x, y, z]. SciPy wants [x, y, z, w]:
    q_RtoW = np.array([
        robot_quat_W[1],
        robot_quat_W[2],
        robot_quat_W[3],
        robot_quat_W[0]
    ])
    R_RtoW = R.from_quat(q_RtoW).as_matrix()   # 3×3: R‐frame axes → world‐frame axes
    T_RtoW = np.eye(4)
    T_RtoW[:3, :3] = R_RtoW
    T_RtoW[:3, 3] = robot_pos_W              # robot_pos_W is [x, y, z] in world

    # 2) Build (RB→gear_desired) as a 4×4:
    #    desired_quat_G_in_RB is [w, x, y, z], so reorder for SciPy:
    q_RBtoG = np.array([
        desired_quat_G_in_RB[1],
        desired_quat_G_in_RB[2],
        desired_quat_G_in_RB[3],
        desired_quat_G_in_RB[0]
    ])
    R_RBtoG = R.from_quat(q_RBtoG).as_matrix()  # 3×3: G‐frame axes → RB‐frame axes
    T_RBtoG = np.eye(4)
    T_RBtoG[:3, :3] = R_RBtoG
    T_RBtoG[:3, 3] = desired_pos_G_in_RB       # [x, y, z] in RB

    # 3) Build the pure “R→RB” rotation (180° about Z, no translation):
    Rz180 = R.from_euler('z', 180, degrees=True).as_matrix()  # 3×3
    T_RtoRB = np.eye(4)
    T_RtoRB[:3, :3] = Rz180
    # T_RtoRB[:3, 3] stays at zero

    # 4) Now form (R→G) = (R→RB) @ (RB→G):
    T_RtoG = T_RtoRB @ T_RBtoG

    # 5) Finally combine robot→world with our R→G to get world→gear_new:
    T_WtoG_new = T_RtoW @ T_RtoG

    # 6) Extract new translation + quaternion (in [w,x,y,z]) for world:
    new_pos_W = T_WtoG_new[:3, 3].tolist()
    new_R_W = T_WtoG_new[:3, :3]
    q_xyzw = R.from_matrix(new_R_W).as_quat()  # SciPy gives [x, y, z, w]
    new_quat_W = [
        float(q_xyzw[3]),  # w
        float(q_xyzw[0]),  # x
        float(q_xyzw[1]),  # y
        float(q_xyzw[2])   # z
    ]

    # 7) Set the gear’s world pose:
    gear_prim = SingleXFormPrim('/World/gear_assembly_frame')
    gear_prim.set_world_pose(position=new_pos_W, orientation=new_quat_W)


def set_gear_at_desired_wrt_robot(
    robot_prim_name: str,
    gear_prim_name: str,
    desired_pos_G_in_R:   list[float],    # [x, y, z] of gear relative to RB‐frame
    desired_quat_G_in_R:  list[float],    # [w, x, y, z] of gear orientation in RB‐frame
    flip_final_pose: bool = False,
):
    """
    Places gear_prim_name so that its pose in R‐frame.

    It matches(desired_pos_G_in_R, desired_quat_G_in_R).
    """
    # 1) Read robot’s world pose (i.e. robot→world):
    robot_prim = SingleXFormPrim(robot_prim_name)
    robot_pos_W, robot_quat_W = robot_prim.get_world_pose()
    # robot_quat_W is [w, x, y, z]. SciPy wants [x, y, z, w]:
    q_RtoW = np.array([
        robot_quat_W[1],
        robot_quat_W[2],
        robot_quat_W[3],
        robot_quat_W[0]
    ])
    R_RtoW = R.from_quat(q_RtoW).as_matrix()   # 3×3: R‐frame axes → world‐frame axes
    T_RtoW = np.eye(4)
    T_RtoW[:3, :3] = R_RtoW
    T_RtoW[:3, 3] = robot_pos_W              # robot_pos_W is [x, y, z] in world

    # 2) Build (RB→gear_desired) as a 4×4:
    #    desired_quat_G_in_RB is [w, x, y, z], so reorder for SciPy:
    q_RtoG = np.array([
        desired_quat_G_in_R[1],
        desired_quat_G_in_R[2],
        desired_quat_G_in_R[3],
        desired_quat_G_in_R[0]
    ])
    R_RtoG = R.from_quat(q_RtoG).as_matrix()  # 3×3: G‐frame axes → RB‐frame axes
    T_RtoG = np.eye(4)
    T_RtoG[:3, :3] = R_RtoG
    T_RtoG[:3, 3] = desired_pos_G_in_R       # [x, y, z] in RB

    # 5) Finally combine robot→world with our R→G to get world→gear_new:
    T_WtoG_new = T_RtoW @ T_RtoG

    if flip_final_pose:
        Rx180 = R.from_euler('x', 180, degrees=True).as_matrix()  # 3×3
        flip_matrix = np.eye(4)
        flip_matrix[:3, :3] = Rx180
        T_WtoG_new = T_WtoG_new @ flip_matrix

    # 6) Extract new translation + quaternion (in [w,x,y,z]) for world:
    new_pos_W = T_WtoG_new[:3, 3].tolist()
    new_R_W = T_WtoG_new[:3, :3]
    q_xyzw = R.from_matrix(new_R_W).as_quat()  # SciPy gives [x, y, z, w]
    new_quat_W = [
        float(q_xyzw[3]),  # w
        float(q_xyzw[0]),  # x
        float(q_xyzw[1]),  # y
        float(q_xyzw[2])   # z
    ]

    # 7) Set the gear’s world pose:
    gear_prim = SingleXFormPrim(gear_prim_name)
    gear_prim.set_world_pose(position=new_pos_W, orientation=new_quat_W)


world = World()
stage = world.stage
robot = Articulation(prim_paths_expr='/World/ur10e_robotiq2f_140_ROS')
prim = SingleXFormPrim('/World/gear_assembly_frame')
prim.set_world_pose(position=[0, 0, 0], orientation=[1, 0, 0, 0])

robot.set_joint_position_targets(
    # Same pose that we use in real.
    positions=[
        2.6342735290,
        -0.875958995,
        1.55104095,
        -2.29230751,
        -1.56029826,
        -2.0387018
    ],
    joint_names=[
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint'
    ]
)

# Gear assembly setup make sure robot and gear stand are correctly placed.
# set_gear_stand_at_desired_wrt_robot(
#     [0.95251, -0.36292, -0.11165],
#     [0.71258, -0.0016871, 0.018713, -0.70133])
set_gear_stand_at_desired_wrt_robot(
    [0.92738, -0.37777, -0.10929],
    [0.71283, -0.011585, 0.0056236, -0.70122])

# Now make sure the gear is placed correctly in the gripper fingers.
desired_position_of_gear_wrt_robotiq_base = [0.0, 0.0, 0.23]  # 15 cm in Z offset
desired_orientation_of_gear_wrt_robotiq_base = [1, 0, 0, 0]

# Actual gear placement so that its between the grippers.
set_gear_at_desired_wrt_robot(
    '/World/ur10e_robotiq2f_140_ROS/ur10e_robotiq2f_140/ee_link/robotiq_base_link',
    '/World/gear_large',
    desired_position_of_gear_wrt_robotiq_base,
    desired_orientation_of_gear_wrt_robotiq_base,
    flip_final_pose=True
)
