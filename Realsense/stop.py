# Pinky = stop

import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np

mp_Hands = mp.solutions.hands
hands = mp_Hands.Hands(static_image_mode=False,
                      max_num_hands=1, # Change for only a certain amount
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pipeline = rs.pipeline()
align_to = rs.stream.color
align = rs.align(align_to)

p_finger_Coord = [(8, 6), (11, 10), (16, 14)] # all fingers EXCEPT the thumb and pinky, as you can flip off with the thumb
pinky_Coord = (19, 18) # pinky coords - with 19 being the CENTER of the pinky

config = rs.config()

config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 60) # This is for depth
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60) # This is for color

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()

align_to = rs.stream.color
align = rs.align(align_to)

framePinky = 0 # how many frames the pinky finger is visible for
frameThresh = 5 # frames the finger should be visible for threshold

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
                cv2.circle(img, point, 3, (255, 0, 0), cv2.FILLED)
            pinkyAboveCount = 0 # how many fingers are below the middle finger
            for coordinate in p_finger_Coord:
                if handList[pinky_Coord[0]][1] + 38.5  <  handList[coordinate[0]][1] or handList[pinky_Coord[0]][1] + 8.5  <  handList[coordinate[0]][1]:
                    pinkyAboveCount += 1
            if(pinkyAboveCount >= 3): # if the amount of fingers (not the thumb) is greater or equal to 3 (index, ring, pinky)
                framePinky += 1
                if(framePinky > frameThresh): # if the finger is there for longer than *frameThresh* frames
                    cv2.putText(img, "stp", (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)
            else: # reset frames if not pinky finger up
                framePinky = 0

            print(pinkyAboveCount) # debug statement

    cv2.imshow("MF", img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break
    

