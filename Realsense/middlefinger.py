from itertools import count
from queue import Empty
import cv2
import mediapipe as mp
import time
import pyrealsense2 as rs
import numpy as np

mp_Hands = mp.solutions.hands
hands = mp_Hands.Hands(static_image_mode=False,
                      max_num_hands=1, # Change for only a certain amount
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
pipeline = rs.pipeline()
align_to = rs.stream.color
align = rs.align(align_to)

finger_Coord = [(8, 6), (16, 14), (20, 18)]
thumb_Coord = (4,2)
middle_Coord = (12, 10)

config = rs.config()

config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 60) #This is for depth
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60) #This is for color

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

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    img = color_image # Change to "color_image for normal"
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    multiLandMarks = results.multi_hand_landmarks
    if multiLandMarks:
        handList = []
    if results.multi_hand_landmarks is not None:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mp_Hands.HAND_CONNECTIONS)
            for idx, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                handList.append((cx, cy))
            for point in handList:
                cv2.circle(img, point, 10, (255, 255, 0), cv2.FILLED)
            upCount = 0
            for coordinate in finger_Coord:
                if handList[middle_Coord[0]][1] + 125  <  handList[coordinate[1]][1]:
                    cv2.putText(img, "middle", (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)
                    # upCount += 1
            #     if handList[coordinate[0]][1] < handList[coordinate[1]][1]: # if the 0th index of a coordinate in finger_coords < 1st index of a coordinate in finger_cords (if a finger is raised)
            #         upCount += 1
            # if handList[thumb_Coord[0]][0] > handList[thumb_Coord[1]][0]:
            #     upCount += 1
            # cv2.putText(img, str(upCount), (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)
            print(handList[12])
            # time.sleep(1)

    cv2.imshow("Counting number of fingers", img)
    cv2.waitKey(1)