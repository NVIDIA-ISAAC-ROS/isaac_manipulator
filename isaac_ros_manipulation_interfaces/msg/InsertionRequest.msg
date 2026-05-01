# Header data
std_msgs/Header header
# Goal pose
geometry_msgs/PoseStamped pose
# Enum detailed below of request type
uint8 request_type

# Request type constants
uint8 NEW_INSERTION = 0
uint8 CANCEL_INSERTION = 1