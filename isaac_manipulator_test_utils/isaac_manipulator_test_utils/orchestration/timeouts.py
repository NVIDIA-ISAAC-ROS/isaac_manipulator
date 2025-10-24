# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Centralized timeout constants for behavior tests."""

# Server configuration timeouts for action/service servers
DEFAULT_SERVER_TIMEOUT_CONFIG = {
    'startup_server_timeout_sec': 30.0,    # Time to wait for server startup
    'runtime_retry_timeout_sec': 10.0,     # Retry timeout during execution
    'server_check_interval_sec': 5.0       # Interval between server checks
}

# Common test timeouts
BEHAVIOR_TIMEOUT = 30.0    # Maximum time for behavior execution
SPIN_TIMEOUT = 0.1         # ROS spin timeout for non-blocking operations
LOG_INTERVAL = 5.0         # Interval for periodic logging

# Node startup delays
MINIMAL_STARTUP_DELAY = 1.0         # Basic test setup delay
EXTERNAL_NODE_STARTUP_DELAY = 2.0   # Delay when launching external nodes


def get_node_startup_delay(has_external_nodes: bool = False) -> float:
    """Get startup delay: minimal for basic tests, longer when launching external nodes."""
    return EXTERNAL_NODE_STARTUP_DELAY if has_external_nodes else MINIMAL_STARTUP_DELAY
