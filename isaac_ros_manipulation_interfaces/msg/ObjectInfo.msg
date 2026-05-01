# Unique Identifier for the object
int32 object_id

# 2D detection data associated with object
vision_msgs/Detection2D detection_2d

# 3D detection data associated with object
vision_msgs/Detection3D detection_3d

# Segmentation Mask
sensor_msgs/Image segmentation_mask
bool has_segmentation_mask

# Mesh file path
string mesh_file_path

# Name of the object
string name
