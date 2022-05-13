import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime

# Configure depth and color streams...
# ...from Camera 1
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device('039322061250')
config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# ...from Camera 2
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device('127122065043')
config_2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


# Start streaming from both cameras
pipeline_1.start(config_1)
pipeline_2.start(config_2)
    
# Camera 1
# Wait for a coherent pair of frames: depth and color
frames_1 = pipeline_1.wait_for_frames()
depth_frame_1 = frames_1.get_depth_frame()
color_frame_1 = frames_1.get_color_frame()

# Convert images to numpy arrays
depth_image_1 = np.asanyarray(depth_frame_1.get_data())
color_image_1 = np.asanyarray(color_frame_1.get_data())

# Camera 2
# Wait for a coherent pair of frames: depth and color
frames_2 = pipeline_2.wait_for_frames()
depth_frame_2 = frames_2.get_depth_frame()
color_frame_2 = frames_2.get_color_frame()

# Convert images to numpy arrays
depth_image_2 = np.asanyarray(depth_frame_2.get_data())
color_image_2 = np.asanyarray(color_frame_2.get_data())

# Save images 
# Save images 
dt = datetime.now()
dt = dt.replace(microsecond=0) 
dt = str(dt)
dt = dt.replace(" ", "_")
dt = dt.replace(":", "_")
dt = dt.replace("-", "_")
cv2.imwrite("./Images/Time_" + str(dt) + "_Top.jpg", color_image_1)
cv2.imwrite("./Images/Time_" + str(dt) + "_Front.jpg", color_image_2)


# Stop streaming
pipeline_1.stop()
pipeline_2.stop()