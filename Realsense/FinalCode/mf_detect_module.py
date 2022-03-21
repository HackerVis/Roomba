import cv2
import mediapipe as mp


mp_Hands = mp.solutions.hands
hands = mp_Hands.Hands(static_image_mode=False,
                      max_num_hands=1, # Change for only a certain amount
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

finger_Coord = [(8, 6), (16, 14), (20, 18)] # all fingers EXCEPT the thumb, as you can flip off with the thumb
middle_Coord = (11, 10) # middle finger coords - with 11 being the CENTER of the middle finger

# middle finger - returns a boolean and frame
def middleFinger(img): # middleFinger(realsense input)
    global mp_Hands, hands, mpDraw, finger_Coord, middle_Coord
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
            middleAboveCount = 0 # how many fingers are above the middle finger
            for coordinate in finger_Coord:
                if handList[middle_Coord[0]][1] + 38.5  <  handList[coordinate[0]][1] or handList[middle_Coord[0]][1] + 8.5  <  handList[coordinate[0]][1]:
                    middleAboveCount += 1
            if(middleAboveCount >= 3): # if the amount of fingers (not the thumb) is greater or equal to 3 (index, ring, pinky)
                print("True")
                return img
            else: # reset frames if not middle finger up
                return False
    else:
        return False
