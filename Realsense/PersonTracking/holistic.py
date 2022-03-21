import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cap.isOpened():
    success, imgHol = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the imgHol as not writeable to
    # pass by reference.
    imgHol.flags.writeable = False
    imgHol = cv2.cvtColor(imgHol, cv2.COLOR_BGR2RGB)
    res = holistic.process(imgHol)

    # Draw landmark annotation on the imgHol.
    imgHol.flags.writeable = True
    imgHol = cv2.cvtColor(imgHol, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        imgHol,
        res.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        imgHol,
        res.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    # Flip the imgHol horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Holistic', cv2.flip(imgHol, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()