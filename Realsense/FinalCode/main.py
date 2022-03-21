import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
from mf_detect_module import middleFinger

config = rs.config()

config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 60) # This is for depth
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60) # This is for color

pipeline = rs.pipeline()
profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()

align_to = rs.stream.color
align = rs.align(align_to)

while True:
    frames = pipeline.wait_for_frames()

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()
    
    if not aligned_depth_frame or not color_frame:
        continue
 
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    middleF = middleFinger(color_image) # check if middle finger is shown
    if(middleF is not False):
        img = middleF # Change to "color_image for normal image"
        print(True)
    else:
        img = color_image

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break